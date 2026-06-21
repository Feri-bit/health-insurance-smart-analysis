# ============================================================
# Exercise 5 - Health Insurance Smart Analysis
# تحلیل هوشمند خدمات بیمه سلامت و پیشنهاد شعبه مناسب
# Author: Farzan Asadi Shekofti
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 0. Global Settings
# ============================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

DATASET_SIZES = [300, 3000, 3000000]

# روی لپ‌تاپ 16GB RAM بهتر است داده بزرگ نمونه‌گیری شود.
LARGE_DATA_THRESHOLD = 100000
MAX_ROWS_FOR_LARGE_SKLEARN = 100000
MAX_ROWS_FOR_LARGE_KERAS = 60000
MAX_ROWS_FOR_AUTOENCODER = 50000

MAX_SAMPLE_FOR_CLUSTER_METRICS = 5000
MAX_ROWS_FOR_HEAVY_CLUSTERING = 5000
MAX_ROWS_FOR_BEST_K = 15000
MAX_ROWS_FOR_CROSS_VALIDATION = 15000

N_CLUSTERS = 4
K_VALUES = range(2, 9)


# ============================================================
# 1. Dataset Generation
# ============================================================

def generate_health_insurance_dataset(n_samples, random_state=42):
    rng = np.random.default_rng(random_state)

    branch_ids = np.array([f"BR-{i:03d}" for i in range(1, 21)])
    branch_quality_map = {b: rng.uniform(50, 95) for b in branch_ids}
    branch_base_workload_map = {b: rng.uniform(20, 90) for b in branch_ids}

    branch_id = rng.choice(branch_ids, size=n_samples)
    branch_quality = np.array([branch_quality_map[b] for b in branch_id])
    branch_base_workload = np.array([branch_base_workload_map[b] for b in branch_id])

    age = rng.integers(18, 85, size=n_samples)

    annual_visits = rng.poisson(4, size=n_samples)
    annual_visits = np.clip(annual_visits, 0, 30)

    service_types = np.array(["visit", "medicine", "lab", "hospital", "dentistry"])
    insurance_types = np.array(["basic", "supplementary", "special"])

    service_type = rng.choice(
        service_types,
        size=n_samples,
        p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    insurance_type = rng.choice(
        insurance_types,
        size=n_samples,
        p=[0.55, 0.35, 0.10]
    )

    distance_to_branch = rng.uniform(0.5, 40, size=n_samples)

    travel_time = (
        distance_to_branch * rng.uniform(1.8, 3.5, size=n_samples)
        + rng.normal(0, 5, size=n_samples)
    )
    travel_time = np.clip(travel_time, 3, 180)

    branch_workload = branch_base_workload + rng.normal(0, 10, size=n_samples)
    branch_workload = np.clip(branch_workload, 5, 100)

    service_workload_factor = (
        (service_type == "hospital") * 18
        + (service_type == "dentistry") * 10
        + (service_type == "lab") * 6
    )

    waiting_time = (
        branch_workload * rng.uniform(0.35, 1.0, size=n_samples)
        + service_workload_factor
        + rng.normal(0, 8, size=n_samples)
    )
    waiting_time = np.clip(waiting_time, 1, 180)

    service_base_cost = {
        "visit": 700_000,
        "medicine": 1_200_000,
        "lab": 2_500_000,
        "hospital": 18_000_000,
        "dentistry": 6_000_000
    }

    insurance_discount = {
        "basic": 0.15,
        "supplementary": 0.35,
        "special": 0.55
    }

    base_cost = np.array([service_base_cost[s] for s in service_type], dtype=float)
    discount = np.array([insurance_discount[i] for i in insurance_type], dtype=float)

    health_risk = (
        age * 0.03
        + annual_visits * 0.25
        + (service_type == "hospital") * 3.0
        + (service_type == "dentistry") * 1.5
        + rng.normal(0, 1, size=n_samples)
    )

    service_cost = (
        base_cost
        * (1 + health_risk * 0.08)
        * (1 - discount)
        + rng.normal(0, 500_000, size=n_samples)
    )
    service_cost = np.clip(service_cost, 100_000, None)

    branch_score = (
        branch_quality
        - distance_to_branch * 0.55
        - travel_time * 0.12
        - waiting_time * 0.20
        - branch_workload * 0.10
        + discount * 12
        + rng.normal(0, 4, size=n_samples)
    )
    branch_score = np.clip(branch_score, 0, 100)

    satisfaction_score = (
        0.55 * branch_score
        + 35
        - service_cost / 3_500_000
        - waiting_time * 0.07
        - travel_time * 0.04
        + rng.normal(0, 5, size=n_samples)
    )
    satisfaction_score = np.clip(satisfaction_score, 0, 100)

    suitable_branch = (
        (satisfaction_score >= 60)
        & (branch_score >= 50)
        & (waiting_time <= 75)
        & (travel_time <= 90)
    ).astype(int)

    satisfaction_level = np.where(
        satisfaction_score >= 70,
        "high",
        np.where(satisfaction_score >= 45, "medium", "low")
    )

    risk_level = np.where(
        (annual_visits >= 8) | (service_cost > np.percentile(service_cost, 75)),
        "high",
        "low"
    )

    positive_comments = np.array([
        "خدمات سریع و کارکنان خوش برخورد بودند",
        "شعبه نزدیک بود و زمان انتظار کم بود",
        "پوشش بیمه مناسب و فرآیند ساده بود",
        "از کیفیت پاسخگویی و سرعت کار رضایت داشتم"
    ], dtype=object)

    neutral_comments = np.array([
        "خدمات معمولی و قابل قبول بود",
        "زمان انتظار متوسط بود",
        "فرآیند انجام کار قابل قبول بود",
        "کیفیت خدمات نه خیلی خوب و نه خیلی بد بود"
    ], dtype=object)

    negative_comments = np.array([
        "زمان انتظار زیاد و شعبه شلوغ بود",
        "فاصله شعبه زیاد بود و رضایت نداشتم",
        "هزینه بالا بود و پاسخگویی ضعیف بود",
        "فرآیند اداری طولانی و خسته کننده بود"
    ], dtype=object)

    user_comment = np.empty(n_samples, dtype=object)

    high_mask = satisfaction_score >= 70
    mid_mask = (satisfaction_score < 70) & (satisfaction_score >= 45)
    low_mask = satisfaction_score < 45

    user_comment[high_mask] = rng.choice(positive_comments, size=high_mask.sum())
    user_comment[mid_mask] = rng.choice(neutral_comments, size=mid_mask.sum())
    user_comment[low_mask] = rng.choice(negative_comments, size=low_mask.sum())

    return pd.DataFrame({
        "age": age,
        "annual_visits": annual_visits,
        "distance_to_branch": distance_to_branch,
        "travel_time": travel_time,
        "service_cost": service_cost,
        "service_type": service_type,
        "insurance_type": insurance_type,
        "branch_id": branch_id,
        "branch_quality": branch_quality,
        "branch_workload": branch_workload,
        "waiting_time": waiting_time,
        "satisfaction_score": satisfaction_score,
        "user_comment": user_comment,
        "suitable_branch": suitable_branch,
        "branch_score": branch_score,
        "satisfaction_level": satisfaction_level,
        "risk_level": risk_level
    })


# ============================================================
# 2. Feature Settings
# ============================================================

BASE_NUMERIC_FEATURES = [
    "age",
    "annual_visits",
    "distance_to_branch",
    "travel_time",
    "branch_quality",
    "branch_workload",
    "waiting_time",
    "branch_score"
]

CATEGORICAL_FEATURES = [
    "service_type",
    "insurance_type",
    "branch_id"
]

TEXT_FEATURE = "user_comment"

REGRESSION_TARGETS = [
    "service_cost",
    "satisfaction_score",
    "waiting_time",
    "branch_score"
]

CLASSIFICATION_TARGETS = [
    "suitable_branch",
    "satisfaction_level",
    "risk_level"
]


def limit_rows(dataset_size, max_rows):
    if dataset_size > LARGE_DATA_THRESHOLD:
        return max_rows, "sampled"
    return dataset_size, "full"


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_numeric_features_for_target(target):
    return [col for col in BASE_NUMERIC_FEATURES if col != target]


def build_sklearn_preprocessor(numeric_features=None):
    if numeric_features is None:
        numeric_features = BASE_NUMERIC_FEATURES

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )


def get_classification_average(y):
    unique_values = set(pd.Series(y).dropna().unique())

    # فقط وقتی برچسب‌ها واقعاً 0 و 1 باشند، binary حساب می‌کنیم
    if unique_values.issubset({0, 1, 0.0, 1.0, True, False}):
        return "binary"

    # برای برچسب‌های متنی مثل high/low یا high/medium/low
    # از weighted استفاده می‌کنیم
    return "weighted"


# ============================================================
# 3. Sklearn Regression
# ============================================================

def evaluate_sklearn_regression(df, dataset_size, run_mode, target):
    numeric_features = get_numeric_features_for_target(target)
    X = df[numeric_features + CATEGORICAL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression + Ridge": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0))
        ]),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01),
        "Decision Tree Regression": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=60, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "KNN Regression": KNeighborsRegressor(n_neighbors=7)
    }

    rows = []
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_sklearn_preprocessor(numeric_features)),
            ("model", model)
        ])

        start = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        overfit_gap = train_r2 - test_r2

        rows.append({
            "dataset_size": dataset_size,
            "rows_used": len(df),
            "run_mode": run_mode,
            "target": target,
            "algorithm": name,
            "learning_type": "Supervised Regression",
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "overfit_gap_r2": overfit_gap,
            "time_seconds": elapsed,
            "overfitting_note": "possible overfitting" if overfit_gap > 0.10 else "no strong overfitting"
        })

    return rows


# ============================================================
# 4. Sklearn Classification
# ============================================================

def evaluate_sklearn_classification(df, dataset_size, run_mode, target):
    X = df[BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target]
    stratify_y = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=60, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=7),
        "SVM Classifier": SVC(kernel="rbf")
    }

    rows = []
    avg = get_classification_average(y)

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_sklearn_preprocessor(BASE_NUMERIC_FEATURES)),
            ("model", model)
        ])

        start = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        acc_gap = train_acc - test_acc

        if avg == "binary":
            precision = precision_score(y_test, y_test_pred, zero_division=0)
            recall = recall_score(y_test, y_test_pred, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)
        else:
            precision = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

        rows.append({
            "dataset_size": dataset_size,
            "rows_used": len(df),
            "run_mode": run_mode,
            "target": target,
            "algorithm": name,
            "learning_type": "Supervised Classification",
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "accuracy_gap": acc_gap,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
            "time_seconds": elapsed,
            "overfitting_note": "possible overfitting" if acc_gap > 0.10 else "no strong overfitting"
        })

    return rows


# ============================================================
# 5. PCA, Clustering, Best K
# ============================================================

def get_processed_matrix_for_unsupervised(df):
    X = df[BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_sklearn_preprocessor(BASE_NUMERIC_FEATURES)
    return preprocessor.fit_transform(X).astype("float32")


def safe_cluster_metrics(X, labels):
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return "-", "-", "-"
    if -1 in unique_labels and len(unique_labels) <= 2:
        return "-", "-", "-"

    if len(X) > MAX_SAMPLE_FOR_CLUSTER_METRICS:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(np.arange(len(X)), size=MAX_SAMPLE_FOR_CLUSTER_METRICS, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels

    if len(set(labels_sample)) < 2:
        return "-", "-", "-"

    try:
        sil = float(silhouette_score(X_sample, labels_sample))
    except Exception:
        sil = "-"

    try:
        ch = float(calinski_harabasz_score(X_sample, labels_sample))
    except Exception:
        ch = "-"

    try:
        db = float(davies_bouldin_score(X_sample, labels_sample))
    except Exception:
        db = "-"

    return sil, ch, db


def evaluate_pca_and_clustering(df, dataset_size, run_mode):
    X_processed = get_processed_matrix_for_unsupervised(df)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_processed).astype("float32")

    rows = [{
        "dataset_size": dataset_size,
        "rows_used": len(df),
        "run_mode": run_mode,
        "algorithm": "PCA",
        "learning_type": "Dimensionality Reduction",
        "n_components": 2,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "silhouette_score": "-",
        "calinski_harabasz_score": "-",
        "davies_bouldin_score": "-",
        "n_clusters": "-",
        "noise_ratio": "-",
        "time_seconds": "-",
        "bic": "-",
        "aic": "-",
        "note": "PCA linear dimensionality reduction"
    }]

    light_models = {
        "K-Means": KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10),
        "MiniBatchKMeans": MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, batch_size=1024),
        "BIRCH": Birch(n_clusters=N_CLUSTERS),
        "Gaussian Mixture Model": GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE)
    }

    heavy_models = {
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=N_CLUSTERS),
        "DBSCAN": DBSCAN(eps=1.2, min_samples=10)
    }

    for name, model in light_models.items():
        start = time.time()
        if name == "Gaussian Mixture Model":
            labels = model.fit_predict(X_pca)
            bic = float(model.bic(X_pca))
            aic = float(model.aic(X_pca))
        else:
            labels = model.fit_predict(X_pca)
            bic = "-"
            aic = "-"
        elapsed = time.time() - start

        sil, ch, db = safe_cluster_metrics(X_pca, labels)

        rows.append({
            "dataset_size": dataset_size,
            "rows_used": len(df),
            "rows_used_for_this_algorithm": len(df),
            "run_mode": run_mode,
            "algorithm": name,
            "learning_type": "Unsupervised Clustering",
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db,
            "noise_ratio": float(np.mean(labels == -1)) if -1 in labels else 0.0,
            "time_seconds": elapsed,
            "bic": bic,
            "aic": aic,
            "note": "Clustering on PCA space"
        })

    for name, model in heavy_models.items():
        if len(X_pca) > MAX_ROWS_FOR_HEAVY_CLUSTERING:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(np.arange(len(X_pca)), size=MAX_ROWS_FOR_HEAVY_CLUSTERING, replace=False)
            X_used = X_pca[idx]
            rows_used_for_model = MAX_ROWS_FOR_HEAVY_CLUSTERING
            note = f"{name} executed on {MAX_ROWS_FOR_HEAVY_CLUSTERING} sampled rows because it is expensive"
        else:
            X_used = X_pca
            rows_used_for_model = len(X_pca)
            note = f"{name} executed on full data"

        start = time.time()
        labels = model.fit_predict(X_used)
        elapsed = time.time() - start

        sil, ch, db = safe_cluster_metrics(X_used, labels)

        rows.append({
            "dataset_size": dataset_size,
            "rows_used": len(df),
            "rows_used_for_this_algorithm": rows_used_for_model,
            "run_mode": run_mode,
            "algorithm": name,
            "learning_type": "Unsupervised Clustering",
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db,
            "noise_ratio": float(np.mean(labels == -1)) if -1 in labels else 0.0,
            "time_seconds": elapsed,
            "bic": "-",
            "aic": "-",
            "note": note
        })

    return rows, X_pca


def evaluate_best_k(df, dataset_size, run_mode):
    X_processed = get_processed_matrix_for_unsupervised(df)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_processed).astype("float32")

    if len(X_pca) > MAX_ROWS_FOR_BEST_K:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(np.arange(len(X_pca)), size=MAX_ROWS_FOR_BEST_K, replace=False)
        X_used = X_pca[idx]
        k_run_mode = "sampled_for_best_k"
    else:
        X_used = X_pca
        k_run_mode = run_mode

    rows = []

    for k in K_VALUES:
        start = time.time()
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_used)
        elapsed = time.time() - start

        sil, ch, db = safe_cluster_metrics(X_used, labels)

        rows.append({
            "dataset_size": dataset_size,
            "rows_used": len(df),
            "rows_used_for_best_k": len(X_used),
            "run_mode": k_run_mode,
            "algorithm": "KMeans Best K Analysis",
            "k": k,
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db,
            "inertia": float(model.inertia_),
            "time_seconds": elapsed
        })

    return rows


def cluster_summary_with_kmeans(df, dataset_size, run_mode):
    X_processed = get_processed_matrix_for_unsupervised(df)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_processed).astype("float32")

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_pca)

    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    summary = df_clustered.groupby("cluster").agg(
        samples_count=("cluster", "count"),
        mean_age=("age", "mean"),
        mean_annual_visits=("annual_visits", "mean"),
        mean_distance_to_branch=("distance_to_branch", "mean"),
        mean_travel_time=("travel_time", "mean"),
        mean_service_cost=("service_cost", "mean"),
        mean_branch_quality=("branch_quality", "mean"),
        mean_branch_workload=("branch_workload", "mean"),
        mean_waiting_time=("waiting_time", "mean"),
        mean_satisfaction_score=("satisfaction_score", "mean"),
        mean_branch_score=("branch_score", "mean"),
        suitable_branch_ratio=("suitable_branch", "mean")
    ).reset_index()

    summary["dataset_size"] = dataset_size
    summary["rows_used"] = len(df)
    summary["run_mode"] = run_mode
    summary["algorithm"] = "KMeans on PCA"
    return summary


# ============================================================
# 6. KNN Branch Recommendation
# ============================================================

def evaluate_knn_branch_recommendation(df, dataset_size, run_mode):
    feature_cols = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["suitable_branch"]

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if pd.Series(y).value_counts().min() >= 2 else None
    )

    preprocessor = build_sklearn_preprocessor(BASE_NUMERIC_FEATURES)
    X_train_processed = preprocessor.fit_transform(X_train).astype("float32")
    X_test_processed = preprocessor.transform(X_test).astype("float32")

    clf = KNeighborsClassifier(n_neighbors=7)

    start = time.time()
    clf.fit(X_train_processed, y_train)
    elapsed = time.time() - start

    y_pred = clf.predict(X_test_processed)

    metric_row = {
        "dataset_size": dataset_size,
        "rows_used": len(df),
        "run_mode": run_mode,
        "algorithm": "KNN Branch Recommendation",
        "purpose": "Recommend suitable branch using similar insured users",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "time_seconds": elapsed
    }

    nn = NearestNeighbors(n_neighbors=7, metric="euclidean")
    nn.fit(X_train_processed)

    sample_count = min(5, len(X_test_processed))
    distances, neighbor_positions = nn.kneighbors(X_test_processed[:sample_count])

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    example_rows = []

    for i in range(sample_count):
        neighbors = train_df.iloc[neighbor_positions[i]].copy()
        suitable_neighbors = neighbors[neighbors["suitable_branch"] == 1]

        if len(suitable_neighbors) > 0:
            branch_scores = suitable_neighbors.groupby("branch_id").agg(
                avg_satisfaction=("satisfaction_score", "mean"),
                avg_branch_score=("branch_score", "mean"),
                count=("branch_id", "count")
            ).reset_index()

            branch_scores["recommendation_score"] = (
                branch_scores["avg_satisfaction"]
                + branch_scores["avg_branch_score"]
                + branch_scores["count"] * 2
            )

            recommended_branch = branch_scores.sort_values(
                "recommendation_score", ascending=False
            ).iloc[0]["branch_id"]
        else:
            recommended_branch = neighbors.sort_values(
                "satisfaction_score", ascending=False
            ).iloc[0]["branch_id"]

        current_user = test_df.iloc[i]

        example_rows.append({
            "dataset_size": dataset_size,
            "run_mode": run_mode,
            "example_id": i + 1,
            "current_branch_id": current_user["branch_id"],
            "recommended_branch_id": recommended_branch,
            "actual_suitable_branch": int(current_user["suitable_branch"]),
            "predicted_suitable_branch": int(y_pred[i]),
            "current_age": current_user["age"],
            "current_service_type": current_user["service_type"],
            "current_insurance_type": current_user["insurance_type"],
            "current_distance_to_branch": current_user["distance_to_branch"],
            "current_waiting_time": current_user["waiting_time"],
            "current_satisfaction_score": current_user["satisfaction_score"]
        })

    return metric_row, example_rows


# ============================================================
# 7. Keras Helpers
# ============================================================

def get_keras_numeric_categorical_text(df, numeric_features):
    numeric = df[numeric_features].astype("float32").to_numpy()

    service_tensor = tf.constant(df["service_type"].fillna("").astype(str).to_numpy(dtype=str), dtype=tf.string)
    insurance_tensor = tf.constant(df["insurance_type"].fillna("").astype(str).to_numpy(dtype=str), dtype=tf.string)
    branch_tensor = tf.constant(df["branch_id"].fillna("").astype(str).to_numpy(dtype=str), dtype=tf.string)
    text_tensor = tf.constant(df[TEXT_FEATURE].fillna("").astype(str).to_numpy(dtype=str), dtype=tf.string)

    return numeric, service_tensor, insurance_tensor, branch_tensor, text_tensor


def build_keras_preprocessing_layers(train_numeric, train_service, train_insurance, train_branch, train_text):
    normalizer = layers.Normalization(name="numeric_normalization")
    normalizer.adapt(train_numeric)

    service_lookup = layers.StringLookup(output_mode="one_hot", name="service_lookup")
    service_lookup.adapt(train_service)

    insurance_lookup = layers.StringLookup(output_mode="one_hot", name="insurance_lookup")
    insurance_lookup.adapt(train_insurance)

    branch_lookup = layers.StringLookup(output_mode="one_hot", name="branch_lookup")
    branch_lookup.adapt(train_branch)

    text_vectorizer = layers.TextVectorization(max_tokens=80, output_mode="tf_idf", name="text_vectorization")
    text_vectorizer.adapt(train_text)

    return normalizer, service_lookup, insurance_lookup, branch_lookup, text_vectorizer


def build_keras_inputs_and_encoded(normalizer, service_lookup, insurance_lookup, branch_lookup, text_vectorizer, numeric_features):
    numeric_input = keras.Input(shape=(len(numeric_features),), name="numeric_input")
    service_input = keras.Input(shape=(), dtype=tf.string, name="service_type_input")
    insurance_input = keras.Input(shape=(), dtype=tf.string, name="insurance_type_input")
    branch_input = keras.Input(shape=(), dtype=tf.string, name="branch_id_input")
    text_input = keras.Input(shape=(), dtype=tf.string, name="text_input")

    x = layers.Concatenate()([
        normalizer(numeric_input),
        service_lookup(service_input),
        insurance_lookup(insurance_input),
        branch_lookup(branch_input),
        text_vectorizer(text_input)
    ])

    return [numeric_input, service_input, insurance_input, branch_input, text_input], x


def keras_input_dict(numeric, service, insurance, branch, text):
    return {
        "numeric_input": numeric,
        "service_type_input": service,
        "insurance_type_input": insurance,
        "branch_id_input": branch,
        "text_input": text
    }


# ============================================================
# 8. Keras Regression for all numeric targets
# ============================================================

def evaluate_keras_regression(df, dataset_size, run_mode, target):
    numeric_features = get_numeric_features_for_target(target)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    train_numeric, train_service, train_insurance, train_branch, train_text = get_keras_numeric_categorical_text(train_df, numeric_features)
    test_numeric, test_service, test_insurance, test_branch, test_text = get_keras_numeric_categorical_text(test_df, numeric_features)

    y_train = train_df[target].astype("float32").to_numpy().reshape(-1, 1)
    y_test = test_df[target].astype("float32").to_numpy().reshape(-1, 1)

    y_scale = 1_000_000.0 if target == "service_cost" else 100.0
    y_train_scaled = y_train / y_scale

    normalizer, service_lookup, insurance_lookup, branch_lookup, text_vectorizer = build_keras_preprocessing_layers(
        train_numeric, train_service, train_insurance, train_branch, train_text
    )

    inputs, x = build_keras_inputs_and_encoded(
        normalizer, service_lookup, insurance_lookup, branch_lookup, text_vectorizer, numeric_features
    )

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)

    output = layers.Dense(1, activation="linear", name="regression_output")(x)
    model = keras.Model(inputs=inputs, outputs=output, name=f"Keras_Regression_{target}")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    if len(df) <= 300:
        epochs = 80
        batch_size = 32
    elif len(df) <= 3000:
        epochs = 60
        batch_size = 64
    else:
        epochs = 25
        batch_size = 512

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]

    start = time.time()
    history = model.fit(
        keras_input_dict(train_numeric, train_service, train_insurance, train_branch, train_text),
        y_train_scaled,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    elapsed = time.time() - start

    y_pred_scaled = model.predict(
        keras_input_dict(test_numeric, test_service, test_insurance, test_branch, test_text),
        verbose=0
    ).reshape(-1)

    y_pred = y_pred_scaled * y_scale
    y_true = y_test.reshape(-1)
    loss_gap = float(history.history["val_loss"][-1] - history.history["loss"][-1])

    return {
        "dataset_size": dataset_size,
        "rows_used": len(df),
        "run_mode": run_mode,
        "target": target,
        "algorithm": "Keras Neural Network Regression",
        "learning_type": "Supervised Deep Learning Regression",
        "test_mae": mean_absolute_error(y_true, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "test_r2": r2_score(y_true, y_pred),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "loss_gap": loss_gap,
        "epochs_used": len(history.history["loss"]),
        "batch_size": batch_size,
        "time_seconds": elapsed,
        "overfitting_note": "possible overfitting" if loss_gap > 0.20 else "no strong overfitting"
    }


# ============================================================
# 9. Keras Classification for all classification targets
# ============================================================

def evaluate_keras_classification(df, dataset_size, run_mode, target):
    numeric_features = BASE_NUMERIC_FEATURES

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[target] if df[target].value_counts().min() >= 2 else None
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[target])
    y_test = label_encoder.transform(test_df[target])

    num_classes = len(label_encoder.classes_)

    train_numeric, train_service, train_insurance, train_branch, train_text = get_keras_numeric_categorical_text(train_df, numeric_features)
    test_numeric, test_service, test_insurance, test_branch, test_text = get_keras_numeric_categorical_text(test_df, numeric_features)

    normalizer, service_lookup, insurance_lookup, branch_lookup, text_vectorizer = build_keras_preprocessing_layers(
        train_numeric, train_service, train_insurance, train_branch, train_text
    )

    inputs, x = build_keras_inputs_and_encoded(
        normalizer, service_lookup, insurance_lookup, branch_lookup, text_vectorizer, numeric_features
    )

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.15)(x)

    if num_classes == 2:
        output = layers.Dense(1, activation="sigmoid", name="classification_output")(x)
        loss = "binary_crossentropy"
        y_train_fit = y_train.astype("float32").reshape(-1, 1)
    else:
        output = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)
        loss = "sparse_categorical_crossentropy"
        y_train_fit = y_train.astype("int32")

    model = keras.Model(inputs=inputs, outputs=output, name=f"Keras_Classifier_{target}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss, metrics=["accuracy"])

    if len(df) <= 300:
        epochs = 80
        batch_size = 32
    elif len(df) <= 3000:
        epochs = 60
        batch_size = 64
    else:
        epochs = 25
        batch_size = 512

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]

    start = time.time()
    history = model.fit(
        keras_input_dict(train_numeric, train_service, train_insurance, train_branch, train_text),
        y_train_fit,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    elapsed = time.time() - start

    y_prob = model.predict(
        keras_input_dict(test_numeric, test_service, test_insurance, test_branch, test_text),
        verbose=0
    )

    if num_classes == 2:
        y_pred = (y_prob.reshape(-1) >= 0.5).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    else:
        y_pred = np.argmax(y_prob, axis=1)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    acc_gap = float(history.history["accuracy"][-1] - history.history["val_accuracy"][-1])

    return {
        "dataset_size": dataset_size,
        "rows_used": len(df),
        "run_mode": run_mode,
        "target": target,
        "classes": list(label_encoder.classes_),
        "algorithm": "Keras Neural Network Classifier",
        "learning_type": "Supervised Deep Learning Classification",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "loss_gap": float(history.history["val_loss"][-1] - history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "accuracy_gap": acc_gap,
        "epochs_used": len(history.history["loss"]),
        "batch_size": batch_size,
        "time_seconds": elapsed,
        "overfitting_note": "possible overfitting" if acc_gap > 0.10 else "no strong overfitting"
    }


# ============================================================
# 10. Keras Autoencoder + KMeans
# ============================================================

def make_keras_feature_matrix(df):
    X = df[BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_sklearn_preprocessor(BASE_NUMERIC_FEATURES)
    return preprocessor.fit_transform(X).astype("float32")


def build_autoencoder(input_dim, latent_dim=2):
    input_layer = keras.Input(shape=(input_dim,), name="autoencoder_input")
    x = layers.Dense(64, activation="relu")(input_layer)
    x = layers.Dense(32, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent_space")(x)
    x = layers.Dense(32, activation="relu")(latent)
    x = layers.Dense(64, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = keras.Model(input_layer, output_layer, name="Keras_Health_Autoencoder")
    encoder = keras.Model(input_layer, latent, name="Keras_Health_Encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return autoencoder, encoder


def evaluate_keras_autoencoder_clustering(df, dataset_size, run_mode):
    if len(df) > MAX_ROWS_FOR_AUTOENCODER:
        df_used = df.sample(n=MAX_ROWS_FOR_AUTOENCODER, random_state=RANDOM_STATE).reset_index(drop=True)
        ae_run_mode = "sampled"
    else:
        df_used = df.copy()
        ae_run_mode = run_mode

    X_processed = make_keras_feature_matrix(df_used)
    autoencoder, encoder = build_autoencoder(input_dim=X_processed.shape[1], latent_dim=2)

    if len(df_used) <= 300:
        epochs = 80
        batch_size = 32
    elif len(df_used) <= 3000:
        epochs = 60
        batch_size = 64
    else:
        epochs = 30
        batch_size = 512

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]

    start = time.time()
    history = autoencoder.fit(
        X_processed,
        X_processed,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    elapsed = time.time() - start

    reconstructed = autoencoder.predict(X_processed, verbose=0)
    latent_features = encoder.predict(X_processed, verbose=0)
    reconstruction_mse = float(np.mean((X_processed - reconstructed) ** 2))

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(latent_features)

    sil, ch, db = safe_cluster_metrics(latent_features, labels)
    loss_gap = float(history.history["val_loss"][-1] - history.history["loss"][-1])

    result = {
        "dataset_size": dataset_size,
        "rows_used": len(df_used),
        "run_mode": ae_run_mode,
        "algorithm": "Keras Autoencoder + KMeans",
        "learning_type": "Unsupervised Deep Learning",
        "latent_dim": 2,
        "n_clusters": N_CLUSTERS,
        "reconstruction_mse": reconstruction_mse,
        "silhouette_score": sil,
        "calinski_harabasz_score": ch,
        "davies_bouldin_score": db,
        "inertia": float(model.inertia_),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "loss_gap": loss_gap,
        "epochs_used": len(history.history["loss"]),
        "batch_size": batch_size,
        "time_seconds": elapsed,
        "overfitting_note": "possible overfitting" if loss_gap > 0.10 else "no strong overfitting"
    }

    df_clustered = df_used.copy()
    df_clustered["cluster"] = labels

    summary = df_clustered.groupby("cluster").agg(
        samples_count=("cluster", "count"),
        mean_age=("age", "mean"),
        mean_annual_visits=("annual_visits", "mean"),
        mean_distance_to_branch=("distance_to_branch", "mean"),
        mean_travel_time=("travel_time", "mean"),
        mean_service_cost=("service_cost", "mean"),
        mean_branch_quality=("branch_quality", "mean"),
        mean_branch_workload=("branch_workload", "mean"),
        mean_waiting_time=("waiting_time", "mean"),
        mean_satisfaction_score=("satisfaction_score", "mean"),
        mean_branch_score=("branch_score", "mean"),
        suitable_branch_ratio=("suitable_branch", "mean")
    ).reset_index()

    summary["dataset_size"] = dataset_size
    summary["rows_used"] = len(df_used)
    summary["run_mode"] = ae_run_mode
    summary["algorithm"] = "Keras Autoencoder + KMeans"

    return result, summary


# ============================================================
# 11. Cross Validation
# ============================================================

def evaluate_cross_validation(df, dataset_size, run_mode):
    if len(df) > MAX_ROWS_FOR_CROSS_VALIDATION:
        df_used = df.sample(n=MAX_ROWS_FOR_CROSS_VALIDATION, random_state=RANDOM_STATE).reset_index(drop=True)
        cv_run_mode = "sampled_for_cv"
    else:
        df_used = df.copy()
        cv_run_mode = run_mode

    rows = []

    target = "service_cost"
    numeric_features = get_numeric_features_for_target(target)
    X_reg = df_used[numeric_features + CATEGORICAL_FEATURES]
    y_reg = df_used[target]

    reg_model = Pipeline([
        ("preprocessor", build_sklearn_preprocessor(numeric_features)),
        ("model", Ridge(alpha=1.0))
    ])

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    start = time.time()
    r2_scores = cross_val_score(reg_model, X_reg, y_reg, cv=kfold, scoring="r2")
    elapsed = time.time() - start

    rows.append({
        "dataset_size": dataset_size,
        "rows_used": len(df),
        "rows_used_for_cv": len(df_used),
        "run_mode": cv_run_mode,
        "task": "Regression",
        "target": target,
        "algorithm": "Ridge Regression",
        "cv_metric": "R2",
        "cv_mean": float(np.mean(r2_scores)),
        "cv_std": float(np.std(r2_scores)),
        "folds": 5,
        "time_seconds": elapsed
    })

    for target in CLASSIFICATION_TARGETS:
        X_cls = df_used[BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y_cls = df_used[target]

        if y_cls.value_counts().min() < 5:
            continue

        cls_model = Pipeline([
            ("preprocessor", build_sklearn_preprocessor(BASE_NUMERIC_FEATURES)),
            ("model", RandomForestClassifier(
                n_estimators=40,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        start = time.time()
        acc_scores = cross_val_score(cls_model, X_cls, y_cls, cv=skfold, scoring="accuracy")
        elapsed = time.time() - start

        rows.append({
            "dataset_size": dataset_size,
            "rows_used": len(df),
            "rows_used_for_cv": len(df_used),
            "run_mode": cv_run_mode,
            "task": "Classification",
            "target": target,
            "algorithm": "Random Forest Classifier",
            "cv_metric": "Accuracy",
            "cv_mean": float(np.mean(acc_scores)),
            "cv_std": float(np.std(acc_scores)),
            "folds": 5,
            "time_seconds": elapsed
        })

    return rows


# ============================================================
# 12. Main Execution
# ============================================================

def main():
    all_regression_rows = []
    all_classification_rows = []
    all_clustering_rows = []
    all_best_k_rows = []
    all_cluster_summaries = []
    all_knn_rows = []
    all_recommendation_examples = []
    all_keras_regression_rows = []
    all_keras_classification_rows = []
    all_keras_autoencoder_rows = []
    all_keras_autoencoder_summaries = []
    all_cv_rows = []

    for dataset_size in DATASET_SIZES:
        rows_used, run_mode = limit_rows(dataset_size, MAX_ROWS_FOR_LARGE_SKLEARN)

        print("=" * 100)
        print(f"Exercise 5 Professional | dataset_size={dataset_size} | rows_used={rows_used} | run_mode={run_mode}")
        print("=" * 100)

        df = generate_health_insurance_dataset(rows_used, random_state=RANDOM_STATE)
        df.to_csv(f"exercise5_health_insurance_dataset_{dataset_size}.csv", index=False, encoding="utf-8-sig")

        for target in REGRESSION_TARGETS:
            print(f"Sklearn Regression | target={target}")
            all_regression_rows.extend(evaluate_sklearn_regression(df, dataset_size, run_mode, target))

        for target in CLASSIFICATION_TARGETS:
            print(f"Sklearn Classification | target={target}")
            all_classification_rows.extend(evaluate_sklearn_classification(df, dataset_size, run_mode, target))

        print("PCA and Clustering")
        clustering_rows, _ = evaluate_pca_and_clustering(df, dataset_size, run_mode)
        all_clustering_rows.extend(clustering_rows)

        print("Best K Analysis")
        all_best_k_rows.extend(evaluate_best_k(df, dataset_size, run_mode))

        print("Cluster Summary")
        all_cluster_summaries.append(cluster_summary_with_kmeans(df, dataset_size, run_mode))

        print("KNN Branch Recommendation")
        knn_row, recommendation_examples = evaluate_knn_branch_recommendation(df, dataset_size, run_mode)
        all_knn_rows.append(knn_row)
        all_recommendation_examples.extend(recommendation_examples)

        print("Cross Validation")
        all_cv_rows.extend(evaluate_cross_validation(df, dataset_size, run_mode))

        keras_rows_used, keras_run_mode = limit_rows(dataset_size, MAX_ROWS_FOR_LARGE_KERAS)
        if keras_rows_used != len(df):
            keras_df = generate_health_insurance_dataset(keras_rows_used, random_state=RANDOM_STATE)
        else:
            keras_df = df.copy()

        for target in REGRESSION_TARGETS:
            print(f"Keras Regression | target={target}")
            all_keras_regression_rows.append(evaluate_keras_regression(keras_df, dataset_size, keras_run_mode, target))

        for target in CLASSIFICATION_TARGETS:
            print(f"Keras Classification | target={target}")
            all_keras_classification_rows.append(evaluate_keras_classification(keras_df, dataset_size, keras_run_mode, target))

        print("Keras Autoencoder + KMeans")
        ae_result, ae_summary = evaluate_keras_autoencoder_clustering(keras_df, dataset_size, keras_run_mode)
        all_keras_autoencoder_rows.append(ae_result)
        all_keras_autoencoder_summaries.append(ae_summary)

    pd.DataFrame(all_regression_rows).to_csv("exercise5_sklearn_regression_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_classification_rows).to_csv("exercise5_sklearn_classification_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_clustering_rows).to_csv("exercise5_sklearn_clustering_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_best_k_rows).to_csv("exercise5_best_k_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_cluster_summaries, ignore_index=True).to_csv("exercise5_sklearn_cluster_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_knn_rows).to_csv("exercise5_knn_branch_recommendation_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_recommendation_examples).to_csv("exercise5_branch_recommendation_examples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_keras_regression_rows).to_csv("exercise5_keras_regression_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_keras_classification_rows).to_csv("exercise5_keras_classification_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_keras_autoencoder_rows).to_csv("exercise5_keras_autoencoder_clustering_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_keras_autoencoder_summaries, ignore_index=True).to_csv("exercise5_keras_autoencoder_cluster_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_cv_rows).to_csv("exercise5_cross_validation_results.csv", index=False, encoding="utf-8-sig")

    print("\nDone. Professional Exercise 5 CSV files created successfully.")
    print("Main outputs:")
    print("- exercise5_sklearn_regression_results.csv")
    print("- exercise5_sklearn_classification_results.csv")
    print("- exercise5_sklearn_clustering_results.csv")
    print("- exercise5_best_k_results.csv")
    print("- exercise5_sklearn_cluster_summary.csv")
    print("- exercise5_knn_branch_recommendation_results.csv")
    print("- exercise5_branch_recommendation_examples.csv")
    print("- exercise5_keras_regression_results.csv")
    print("- exercise5_keras_classification_results.csv")
    print("- exercise5_keras_autoencoder_clustering_results.csv")
    print("- exercise5_keras_autoencoder_cluster_summary.csv")
    print("- exercise5_cross_validation_results.csv")


if __name__ == "__main__":
    main()
