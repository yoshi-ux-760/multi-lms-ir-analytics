
# multi-lms-ir-analytics
python code ETL

"""
Title: Multi-LMS to Google BigQuery ETL and Predictive Learning Analytics Pipeline
Description: Extracts pseudonymized LMS telemetry, engineers SRL engagement features,
             loads tables into Google BigQuery for IR integration, and executes
             ANCOVA and recall-optimized early-warning classification.
"""
import hashlib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score

def pseudonymize_id(raw_id: str, salt: str) -> str:
    """Apply salted SHA-256 hashing to strip PII prior to cloud warehouse ingestion."""
    return hashlib.sha256(f"{salt}:{raw_id}".encode("utf-8")).hexdigest()[:16]

def run_bigquery_learning_analytics(
    log_filepath: str, score_filepath: str, bq_project_id: str,
    bq_dataset: str, salt_key: str, upload_to_bq: bool = False
):
    # 1. Ingest raw LMS logs and strip PII via cryptographic hashing
    logs = pd.read_csv(log_filepath, parse_dates=["timestamp"])
    scores = pd.read_csv(score_filepath)
    logs["student_hash_id"] = logs["raw_student_id"].astype(str).apply(lambda x: pseudonymize_id(x, salt_key))
    scores["student_hash_id"] = scores["raw_student_id"].astype(str).apply(lambda x: pseudonymize_id(x, salt_key))

    # 2. Feature Engineering: Construct longitudinal SRL and engagement metrics
    logs = logs.sort_values(["student_hash_id", "timestamp"])
    logs["date"] = logs["timestamp"].dt.date
    logs["inter_event_hours"] = logs.groupby("student_hash_id")["timestamp"].diff().dt.total_seconds() / 3600.0

    features = logs.groupby(["student_hash_id", "lms_source"]).agg(
        total_events=("event_name", "count"),
        active_days=("date", "nunique"),
        session_regularity_std=("inter_event_hours", "std"),
        quiz_attempts=("is_quiz_attempt", "sum"),
        late_submissions=("is_late_submission", "sum"),
        ai_scaffold_requests=("is_ai_action", "sum"),
        reflection_count=("is_reflection", "sum")
    ).reset_index().fillna({"session_regularity_std": 0.0})

    features["ai_dependency_ratio"] = features["ai_scaffold_requests"] / (features["total_events"] + 1e-5)

    # 3. Merge with baseline/post assessments and define academic risk (post_score < 60)
    analytical_df = scores.drop(columns=["raw_student_id"]).merge(features, on="student_hash_id", how="left").fillna(0)
    analytical_df["gain_score"] = analytical_df["post_score"] - analytical_df["pre_score"]
    analytical_df["at_risk"] = np.where(analytical_df["post_score"] < 60.0, 1, 0)

    # 4. Optional Cloud Warehouse Load: Push clean analytical table to Google BigQuery
    if upload_to_bq:
        client = bigquery.Client(project=bq_project_id)
        table_id = f"{bq_project_id}.{bq_dataset}.fact_student_learning_analytics"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(analytical_df, table_id, job_config=job_config)
        job.result()

    # 5. Inferential Evaluation: Paired t-test & ANCOVA
    t_val, p_val = stats.ttest_rel(analytical_df["post_score"], analytical_df["pre_score"])
    ancova = smf.ols("post_score ~ C(group) + pre_score + active_days + ai_dependency_ratio", data=analytical_df).fit()

    # 6. Early-Warning Risk Classifier (Optimizing Recall to minimize missed at-risk students)
    feature_cols = ["pre_score", "total_events", "active_days", "session_regularity_std",
                    "quiz_attempts", "late_submissions", "ai_scaffold_requests", "ai_dependency_ratio"]
    X, y = analytical_df[feature_cols], analytical_df["at_risk"]
    clf_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", random_state=42))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(clf_pipeline, X, y, cv=cv, scoring={"recall": make_scorer(recall_score), "roc_auc": "roc_auc"})
    return analytical_df, ancova, clf_pipeline
