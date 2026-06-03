from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zlib import crc32

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from fast_tracking_ssq_dataset import FEATURE_COLUMNS, HEAD_FEATURE_COLUMNS, VR_SYSTEM_FEATURE_COL

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception:
    BalancedRandomForestClassifier = None

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

try:
    from imblearn.under_sampling import NearMiss
except Exception:
    NearMiss = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


HEADFEATURES_DIR = ROOT / "headfeatures_data"
OUTPUT_DIR = ROOT / "outputs" / "article_best"
DEFAULT_CONFIG_PATH = ROOT / "configs" / "pipeline_article_best.json"
RANDOM_STATE = 42
N_FOLDS = 7
FEATURE_REDUCTION_METHOD = "pearson_redundancy_filter"
FEATURE_REDUCTION_CORR_METHOD = "pearson"
FEATURE_REDUCTION_THRESHOLD = 0.90
FEATURE_REDUCTION_CACHE: dict[tuple[object, ...], tuple[list[str], pd.DataFrame]] = {}


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lower_bounds_ = np.nanquantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    train_a_path: Path
    train_b_path: Path
    meta_a_path: Path
    meta_b_path: Path
    target_column: str
    classification_type: str
    primary_metric: str
    threshold_strategies: tuple[str, ...]


@dataclass(frozen=True)
class FeatureReductionConfig:
    enabled: bool = True
    method: str = FEATURE_REDUCTION_METHOD
    corr_method: str = FEATURE_REDUCTION_CORR_METHOD
    corr_threshold: float = FEATURE_REDUCTION_THRESHOLD


TASKS: dict[str, TaskSpec] = {
    "sus_binary": TaskSpec(
        name="sus_binary",
        train_a_path=HEADFEATURES_DIR / "HeadFeaturesVSSUSBinary_BuildA.xlsx",
        train_b_path=HEADFEATURES_DIR / "HeadFeaturesVSSUSBinary_BuildB.xlsx",
        meta_a_path=HEADFEATURES_DIR / "HeadFeaturesVSSUSBinary_BuildA_metadata.csv",
        meta_b_path=HEADFEATURES_DIR / "HeadFeaturesVSSUSBinary_BuildB_metadata.csv",
        target_column="sus_not_acceptable_target",
        classification_type="binary",
        primary_metric="f1_positive_mean",
        threshold_strategies=("default_score_threshold", "roc_gmean", "pr_f1", "pr_ideal_distance"),
    ),
    "tlx_binary": TaskSpec(
        name="tlx_binary",
        train_a_path=HEADFEATURES_DIR / "HeadFeaturesVSTLXBinary_BuildA.xlsx",
        train_b_path=HEADFEATURES_DIR / "HeadFeaturesVSTLXBinary_BuildB.xlsx",
        meta_a_path=HEADFEATURES_DIR / "HeadFeaturesVSTLXBinary_BuildA_metadata.csv",
        meta_b_path=HEADFEATURES_DIR / "HeadFeaturesVSTLXBinary_BuildB_metadata.csv",
        target_column="tlx_not_low_target",
        classification_type="binary",
        primary_metric="f1_positive_mean",
        threshold_strategies=("default_score_threshold", "roc_gmean", "pr_f1", "pr_ideal_distance"),
    ),
    "spes_binary": TaskSpec(
        name="spes_binary",
        train_a_path=HEADFEATURES_DIR / "HeadFeaturesVSSPESBinary_BuildA.xlsx",
        train_b_path=HEADFEATURES_DIR / "HeadFeaturesVSSPESBinary_BuildB.xlsx",
        meta_a_path=HEADFEATURES_DIR / "HeadFeaturesVSSPESBinary_BuildA_metadata.csv",
        meta_b_path=HEADFEATURES_DIR / "HeadFeaturesVSSPESBinary_BuildB_metadata.csv",
        target_column="spes_not_high_target",
        classification_type="binary",
        primary_metric="f1_positive_mean",
        threshold_strategies=("default_score_threshold", "roc_gmean", "pr_f1", "pr_ideal_distance"),
    ),
    "ssq_3class": TaskSpec(
        name="ssq_3class",
        train_a_path=HEADFEATURES_DIR / "HeadFeaturesVSSSQ3Class_BuildA.xlsx",
        train_b_path=HEADFEATURES_DIR / "HeadFeaturesVSSSQ3Class_BuildB.xlsx",
        meta_a_path=HEADFEATURES_DIR / "HeadFeaturesVSSSQ3Class_BuildA_metadata.csv",
        meta_b_path=HEADFEATURES_DIR / "HeadFeaturesVSSSQ3Class_BuildB_metadata.csv",
        target_column="ssq_3class_target",
        classification_type="multiclass",
        primary_metric="macro_f1_mean",
        threshold_strategies=("N/A",),
    ),
}


MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "balanced_random_forest": [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
    ],
    "catboost": [
        {"depth": 4, "learning_rate": 0.03, "iterations": 300},
        {"depth": 6, "learning_rate": 0.03, "iterations": 300},
        {"depth": 6, "learning_rate": 0.05, "iterations": 150},
    ],
    "logreg": [
        {"C": 0.25},
        {"C": 1.0},
        {"C": 4.0},
    ],
    "svm": [
        {"kernel": "linear", "C": 0.25},
        {"kernel": "linear", "C": 1.0},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    ],
    "random_forest": [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
    ],
    "xgboost": [
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    ],
    "lightgbm": [
        {"n_estimators": 150, "num_leaves": 15, "learning_rate": 0.05},
        {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.05},
    ],
}

PREPROCESSING_CANDIDATES: dict[str, list[str]] = {
    "balanced_random_forest": ["median_only", "winsor_only"],
    "catboost": ["median_only", "winsor_only"],
    "logreg": ["robust", "standard", "winsor_robust", "power_standard"],
    "svm": ["robust", "standard", "winsor_robust", "power_standard"],
    "random_forest": ["median_only", "winsor_only"],
    "xgboost": ["median_only", "winsor_only"],
    "lightgbm": ["median_only", "winsor_only"],
}

BINARY_IMBALANCE_BY_MODEL: dict[str, list[str]] = {
    "balanced_random_forest": ["internal_balance", "undersample", "smote", "nearmiss"],
    "catboost": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "logreg": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "svm": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "random_forest": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "xgboost": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "lightgbm": ["none", "class_weight", "undersample", "smote", "nearmiss"],
}

MULTICLASS_IMBALANCE_BY_MODEL: dict[str, list[str]] = {
    "balanced_random_forest": ["internal_balance", "undersample", "smote", "nearmiss"],
    "catboost": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "logreg": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "svm": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "random_forest": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "xgboost": ["none", "class_weight", "undersample", "smote", "nearmiss"],
    "lightgbm": ["none", "class_weight", "undersample", "smote", "nearmiss"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the article reproduction pipeline with the selected best "
            "configuration for each questionnaire."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="JSON pipeline config with paths and selected task configurations.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=sorted(TASKS),
        help=(
            "Optional task filter. Can be repeated. "
            f"Available: {', '.join(sorted(TASKS))}."
        ),
    )
    return parser.parse_args()


def load_json_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def task_with_headfeatures_dir(task: TaskSpec, headfeatures_dir: Path) -> TaskSpec:
    return TaskSpec(
        name=task.name,
        train_a_path=headfeatures_dir / task.train_a_path.name,
        train_b_path=headfeatures_dir / task.train_b_path.name,
        meta_a_path=headfeatures_dir / task.meta_a_path.name,
        meta_b_path=headfeatures_dir / task.meta_b_path.name,
        target_column=task.target_column,
        classification_type=task.classification_type,
        primary_metric=task.primary_metric,
        threshold_strategies=task.threshold_strategies,
    )


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pooled_task_dataframe(task: TaskSpec) -> pd.DataFrame:
    df_a = pd.read_excel(task.train_a_path)
    df_b = pd.read_excel(task.train_b_path)
    meta_a = pd.read_csv(task.meta_a_path)
    meta_b = pd.read_csv(task.meta_b_path)

    required_meta = ["score_pid"]
    for column in required_meta:
        if column not in meta_a.columns or column not in meta_b.columns:
            raise ValueError(f"Metadata for {task.name} must contain '{column}'.")

    pooled_a = df_a.copy()
    pooled_a["build"] = "A"
    pooled_a["score_pid"] = meta_a["score_pid"].astype(str).to_numpy()
    pooled_a["row_id"] = [f"A_{i}" for i in range(len(pooled_a))]

    pooled_b = df_b.copy()
    pooled_b["build"] = "B"
    pooled_b["score_pid"] = meta_b["score_pid"].astype(str).to_numpy()
    pooled_b["row_id"] = [f"B_{i}" for i in range(len(pooled_b))]

    pooled = pd.concat([pooled_a, pooled_b], ignore_index=True)
    pooled["score_pid"] = pooled["score_pid"].astype(str)
    pooled[task.target_column] = pd.to_numeric(pooled[task.target_column], errors="raise").astype(int)
    return pooled


def available_models() -> list[str]:
    models: list[str] = []
    for model_name in MODEL_GRIDS:
        if model_name == "balanced_random_forest" and BalancedRandomForestClassifier is None:
            continue
        if model_name == "catboost" and CatBoostClassifier is None:
            continue
        if model_name == "xgboost" and XGBClassifier is None:
            continue
        if model_name == "lightgbm" and LGBMClassifier is None:
            continue
        models.append(model_name)
    return models


def build_preprocessing_steps(preprocess_name: str) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if preprocess_name == "median_only":
        return steps
    if preprocess_name == "winsor_only":
        steps.append(("winsor", Winsorizer(0.01, 0.99)))
        return steps
    if preprocess_name == "standard":
        steps.append(("scaler", StandardScaler()))
        return steps
    if preprocess_name == "robust":
        steps.append(("scaler", RobustScaler()))
        return steps
    if preprocess_name == "winsor_robust":
        steps.append(("winsor", Winsorizer(0.01, 0.99)))
        steps.append(("scaler", RobustScaler()))
        return steps
    if preprocess_name == "power_standard":
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
        steps.append(("scaler", StandardScaler()))
        return steps
    raise ValueError(f"Unsupported preprocessing strategy: {preprocess_name}")


def _safe_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(np.nanstd(x)) == 0.0 or float(np.nanstd(y)) == 0.0:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return abs(value)


def reduce_features_by_pearson(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    task: TaskSpec,
    config: FeatureReductionConfig,
    split_id: int,
    selection_stage: str,
) -> tuple[list[str], pd.DataFrame]:
    cache_key = (
        task.name,
        int(split_id),
        selection_stage,
        bool(config.enabled),
        config.method,
        config.corr_method,
        float(config.corr_threshold),
    )
    cached = FEATURE_REDUCTION_CACHE.get(cache_key)
    if cached is not None:
        selected_cached, rows_cached = cached
        return list(selected_cached), rows_cached.copy()

    available_head_features = [feature for feature in HEAD_FEATURE_COLUMNS if feature in X_train.columns]
    context_features = [VR_SYSTEM_FEATURE_COL] if VR_SYSTEM_FEATURE_COL in X_train.columns else []

    if not config.enabled:
        selected = [feature for feature in FEATURE_COLUMNS if feature in X_train.columns]
        rows = [
            {
                "task": task.name,
                "split_id": int(split_id),
                "selection_stage": selection_stage,
                "feature_reduction_method": "disabled_all_features",
                "corr_method": "N/A",
                "corr_threshold": "N/A",
                "group_id": 1,
                "group_size": len(selected),
                "selected_feature": ";".join(selected),
                "removed_features": "",
                "selected_target_abs_pearson": "N/A",
                "group_features": ";".join(selected),
                "n_selected_features": len(selected),
            }
        ]
        rows_df = pd.DataFrame(rows)
        FEATURE_REDUCTION_CACHE[cache_key] = (list(selected), rows_df.copy())
        return selected, rows_df

    if not available_head_features:
        rows_df = pd.DataFrame()
        FEATURE_REDUCTION_CACHE[cache_key] = (list(context_features), rows_df.copy())
        return context_features, rows_df

    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_train[available_head_features]),
        columns=available_head_features,
        index=X_train.index,
    )
    y_values = pd.to_numeric(y_train, errors="coerce").to_numpy(dtype=float)
    target_abs_corr = {
        feature: _safe_abs_pearson(imputed[feature].to_numpy(dtype=float), y_values)
        for feature in available_head_features
    }
    feature_corr = imputed.corr(method=config.corr_method).abs().fillna(0.0)

    adjacency: dict[str, set[str]] = {feature: set() for feature in available_head_features}
    for i, feature_i in enumerate(available_head_features):
        for feature_j in available_head_features[i + 1 :]:
            if float(feature_corr.loc[feature_i, feature_j]) > float(config.corr_threshold):
                adjacency[feature_i].add(feature_j)
                adjacency[feature_j].add(feature_i)

    visited: set[str] = set()
    selected_head_features: list[str] = []
    rows: list[dict[str, Any]] = []

    for feature in available_head_features:
        if feature in visited:
            continue
        stack = [feature]
        component: list[str] = []
        visited.add(feature)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        group_features = [candidate for candidate in available_head_features if candidate in set(component)]
        selected_feature = sorted(
            group_features,
            key=lambda candidate: (
                target_abs_corr.get(candidate, 0.0),
                -available_head_features.index(candidate),
            ),
            reverse=True,
        )[0]
        selected_head_features.append(selected_feature)
        removed_features = [candidate for candidate in group_features if candidate != selected_feature]
        rows.append(
            {
                "task": task.name,
                "split_id": int(split_id),
                "selection_stage": selection_stage,
                "feature_reduction_method": config.method,
                "corr_method": config.corr_method,
                "corr_threshold": float(config.corr_threshold),
                "group_id": len(rows) + 1,
                "group_size": len(group_features),
                "selected_feature": selected_feature,
                "removed_features": ";".join(removed_features),
                "selected_target_abs_pearson": float(target_abs_corr.get(selected_feature, 0.0)),
                "group_features": ";".join(group_features),
                "n_selected_features": len(selected_head_features) + len(context_features),
            }
        )

    selected_features = [feature for feature in FEATURE_COLUMNS if feature in set(selected_head_features + context_features)]
    for row in rows:
        row["n_selected_features"] = len(selected_features)
    rows_df = pd.DataFrame(rows)
    FEATURE_REDUCTION_CACHE[cache_key] = (list(selected_features), rows_df.copy())
    return selected_features, rows_df


def summarize_feature_selection(split_outputs: list[dict[str, Any]], config: FeatureReductionConfig) -> dict[str, Any]:
    selected_by_split = [item.get("final_selected_features") or item.get("selected_features") for item in split_outputs]
    selected_by_split = [features for features in selected_by_split if features]
    counts = [len(features) for features in selected_by_split]
    if not counts:
        return {
            "feature_reduction_method": "unknown",
            "feature_reduction_corr_method": "unknown",
            "feature_reduction_threshold": np.nan,
            "n_features_mean": np.nan,
            "n_features_min": np.nan,
            "n_features_max": np.nan,
        }
    return {
        "feature_reduction_method": config.method if config.enabled else "disabled_all_features",
        "feature_reduction_corr_method": config.corr_method if config.enabled else "N/A",
        "feature_reduction_threshold": float(config.corr_threshold) if config.enabled else "N/A",
        "n_features_mean": float(np.mean(counts)),
        "n_features_min": int(np.min(counts)),
        "n_features_max": int(np.max(counts)),
    }


def class_weight_binary_dict(y: pd.Series | np.ndarray) -> dict[int, float]:
    y = np.asarray(y, dtype=int)
    counts = pd.Series(y).value_counts().sort_index()
    negative = int(counts.get(0, 0))
    positive = int(counts.get(1, 0))
    if negative == 0 or positive == 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: negative / positive}


def class_weight_multiclass_dict(y: pd.Series | np.ndarray) -> dict[int, float]:
    counts = pd.Series(np.asarray(y, dtype=int)).value_counts().sort_index()
    total = int(counts.sum())
    n_classes = int(len(counts))
    return {int(label): total / (n_classes * int(count)) for label, count in counts.items()}


def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    strategy: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if strategy in {"none", "class_weight", "internal_balance"}:
        return X, y

    if strategy == "undersample":
        counts = y.value_counts()
        min_count = int(counts.min())
        rng = np.random.default_rng(random_state)
        chosen: list[int] = []
        for label in counts.index:
            idx = y.index[y.eq(label)].to_numpy()
            if len(idx) > min_count:
                idx = rng.choice(idx, size=min_count, replace=False)
            chosen.extend(idx.tolist())
        rng.shuffle(chosen)
        return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)

    if strategy == "smote":
        if SMOTE is None:
            raise ImportError("imbalanced-learn is required for SMOTE.")
        min_count = int(y.value_counts().min())
        if min_count < 2:
            return X, y
        sampler = SMOTE(random_state=random_state, k_neighbors=min(5, min_count - 1))
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    if strategy == "nearmiss":
        if NearMiss is None:
            raise ImportError("imbalanced-learn is required for NearMiss.")
        sampler = NearMiss(version=1)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    raise ValueError(f"Unsupported imbalance strategy: {strategy}")


def build_model(
    *,
    task: TaskSpec,
    model_name: str,
    params: dict[str, Any],
    imbalance_strategy: str,
    y_reference: pd.Series,
    random_state: int,
) -> object:
    params = dict(params)
    classification_type = task.classification_type
    if classification_type == "binary":
        weight_dict = class_weight_binary_dict(y_reference)
    else:
        weight_dict = class_weight_multiclass_dict(y_reference)

    if model_name == "logreg":
        solver = "liblinear" if classification_type == "binary" else "lbfgs"
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver=str(params.get("solver", solver)),
            max_iter=3000,
            random_state=random_state,
            class_weight="balanced" if imbalance_strategy == "class_weight" else None,
        )

    if model_name == "svm":
        return SVC(
            C=float(params.get("C", 1.0)),
            kernel=str(params.get("kernel", "linear")),
            gamma=params.get("gamma", "scale"),
            probability=True,
            random_state=random_state,
            class_weight="balanced" if imbalance_strategy == "class_weight" else None,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if params.get("max_depth") is None else int(params.get("max_depth")),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced_subsample" if imbalance_strategy == "class_weight" else None,
        )

    if model_name == "balanced_random_forest":
        if BalancedRandomForestClassifier is None:
            raise ImportError("imbalanced-learn is required for balanced_random_forest.")
        return BalancedRandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if params.get("max_depth") is None else int(params.get("max_depth")),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_state,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            n_jobs=1,
        )

    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is required for catboost experiments.")
        if classification_type == "binary":
            class_weights = None
            if imbalance_strategy == "class_weight":
                class_weights = [weight_dict.get(0, 1.0), weight_dict.get(1, 1.0)]
            return CatBoostClassifier(
                depth=int(params.get("depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.03)),
                iterations=int(params.get("iterations", 300)),
                loss_function="Logloss",
                eval_metric="Logloss",
                verbose=False,
                allow_writing_files=False,
                random_seed=random_state,
                class_weights=class_weights,
            )
        class_weights = None
        if imbalance_strategy == "class_weight":
            labels = sorted(weight_dict)
            class_weights = [weight_dict[label] for label in labels]
        return CatBoostClassifier(
            depth=int(params.get("depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            iterations=int(params.get("iterations", 300)),
            loss_function="MultiClass",
            eval_metric="MultiClass",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_state,
            class_weights=class_weights,
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is required for xgboost experiments.")
        if classification_type == "binary":
            scale_pos_weight = float(weight_dict.get(1, 1.0)) if imbalance_strategy == "class_weight" else 1.0
            return XGBClassifier(
                n_estimators=int(params.get("n_estimators", 150)),
                max_depth=int(params.get("max_depth", 3)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=1,
            )
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            objective="multi:softprob",
            num_class=int(pd.Series(y_reference).nunique()),
            eval_metric="mlogloss",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=1,
        )

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required for lightgbm experiments.")
        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            num_leaves=int(params.get("num_leaves", 15)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            objective="binary" if classification_type == "binary" else "multiclass",
            num_class=None if classification_type == "binary" else int(pd.Series(y_reference).nunique()),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
            class_weight="balanced" if imbalance_strategy == "class_weight" else None,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def fit_pipeline(
    *,
    task: TaskSpec,
    model_name: str,
    model_params: dict[str, Any],
    preprocess_name: str,
    imbalance_strategy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Pipeline:
    X_fit, y_fit = apply_resampling(X_train, y_train, strategy=imbalance_strategy, random_state=random_state)
    model = build_model(
        task=task,
        model_name=model_name,
        params=model_params,
        imbalance_strategy=imbalance_strategy,
        y_reference=y_train,
        random_state=random_state,
    )
    pipeline = Pipeline(build_preprocessing_steps(preprocess_name) + [("model", model)])

    fit_kwargs: dict[str, Any] = {}
    if task.classification_type == "multiclass" and model_name == "xgboost" and imbalance_strategy == "class_weight":
        fit_kwargs["model__sample_weight"] = np.asarray(
            [class_weight_multiclass_dict(y_train)[int(label)] for label in y_fit],
            dtype=float,
        )

    pipeline.fit(X_fit, y_fit, **fit_kwargs)
    return pipeline


def score_binary_model(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, str, float]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float), "predict_proba", 0.5
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores, dtype=float), "decision_function", 0.0
    raise ValueError("Binary model exposes neither predict_proba nor decision_function.")


def threshold_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)


def select_threshold(y_true: np.ndarray, scores: np.ndarray, strategy: str, default_threshold: float) -> float:
    if strategy == "default_score_threshold":
        return float(default_threshold)

    if len(np.unique(y_true)) < 2:
        return float(default_threshold)

    if strategy == "roc_gmean":
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        gmeans = np.sqrt(tpr * (1.0 - fpr))
        finite_mask = np.isfinite(thresholds)
        if finite_mask.any():
            candidate_ix = np.where(finite_mask)[0]
            best_local = int(np.argmax(gmeans[finite_mask]))
            ix = int(candidate_ix[best_local])
        else:
            ix = int(np.argmax(gmeans))
        return float(thresholds[ix])

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return float(default_threshold)

    precision = precision[:-1]
    recall = recall[:-1]
    if strategy == "pr_f1":
        denom = precision + recall
        f1_values = np.divide(
            2 * precision * recall,
            denom,
            out=np.zeros_like(precision),
            where=denom > 0,
        )
        return float(thresholds[int(np.nanargmax(f1_values))])

    if strategy == "pr_ideal_distance":
        distance = np.sqrt((1.0 - precision) ** 2 + (1.0 - recall) ** 2)
        return float(thresholds[int(np.nanargmin(distance))])

    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix_json": json.dumps(cm.tolist()),
    }
    return metrics


def participant_group_vectors(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[int]]:
    labels = sorted(df[target_column].unique().tolist())
    records: list[dict[str, Any]] = []
    for score_pid, group_df in df.groupby("score_pid", sort=True):
        counts = group_df[target_column].value_counts().to_dict()
        row: dict[str, Any] = {"score_pid": score_pid, "n_rows": int(len(group_df))}
        for label in labels:
            row[f"class_{label}_count"] = int(counts.get(label, 0))
        records.append(row)
    return pd.DataFrame(records), labels


def assign_grouped_stratified_folds(df: pd.DataFrame, target_column: str, n_folds: int, seed: int) -> pd.DataFrame:
    groups_df, labels = participant_group_vectors(df, target_column)
    ideal_group_count = len(groups_df) / n_folds
    total_class_counts = np.asarray([groups_df[f"class_{label}_count"].sum() for label in labels], dtype=float)
    ideal_class_counts = total_class_counts / n_folds

    rarity_weights = np.divide(
        1.0,
        np.maximum(total_class_counts, 1.0),
        out=np.ones_like(total_class_counts, dtype=float),
    )

    group_vectors = groups_df[[f"class_{label}_count" for label in labels]].to_numpy(dtype=float)
    rarity_score = group_vectors @ rarity_weights
    major_class = np.argmax(group_vectors, axis=1)

    order_df = groups_df.copy()
    order_df["rarity_score"] = rarity_score
    order_df["major_class"] = major_class
    rng = np.random.default_rng(seed)
    order_df["tie_noise"] = rng.uniform(0, 1e-6, size=len(order_df))
    order_df = order_df.sort_values(
        by=["rarity_score", "n_rows", "major_class", "tie_noise"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    fold_group_counts = np.zeros(n_folds, dtype=float)
    fold_class_counts = np.zeros((n_folds, len(labels)), dtype=float)
    assignments: list[dict[str, Any]] = []

    for row in order_df.itertuples(index=False):
        group_vector = np.asarray([getattr(row, f"class_{label}_count") for label in labels], dtype=float)
        best_fold = None
        best_cost = None
        for fold_id in range(n_folds):
            trial_group_counts = fold_group_counts.copy()
            trial_class_counts = fold_class_counts.copy()
            trial_group_counts[fold_id] += 1
            trial_class_counts[fold_id] += group_vector

            class_cost = float(np.sum((trial_class_counts - ideal_class_counts) ** 2))
            group_cost = float(np.sum((trial_group_counts - ideal_group_count) ** 2))
            cost = class_cost + 0.25 * group_cost
            if best_cost is None or cost < best_cost - 1e-12 or (
                math.isclose(cost, best_cost, rel_tol=0, abs_tol=1e-12)
                and trial_group_counts[fold_id] < trial_group_counts[best_fold]
            ):
                best_fold = fold_id
                best_cost = cost

        fold_group_counts[best_fold] += 1
        fold_class_counts[best_fold] += group_vector
        assignments.append(
            {
                "score_pid": row.score_pid,
                "fold_id": int(best_fold),
                "rarity_score": float(row.rarity_score),
                **{f"class_{label}_count": int(getattr(row, f"class_{label}_count")) for label in labels},
            }
        )

    assignment_df = pd.DataFrame(assignments)
    return assignment_df


def fold_diagnostics(df: pd.DataFrame, assignment_df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = df.merge(assignment_df[["score_pid", "fold_id"]], on="score_pid", how="left")
    labels = sorted(df[target_column].unique().tolist())

    diag_rows: list[dict[str, Any]] = []
    for fold_id, fold_df in merged.groupby("fold_id", sort=True):
        row: dict[str, Any] = {
            "fold_id": int(fold_id),
            "participants": int(fold_df["score_pid"].nunique()),
            "rows": int(len(fold_df)),
            "build_A_rows": int((fold_df["build"] == "A").sum()),
            "build_B_rows": int((fold_df["build"] == "B").sum()),
        }
        counts = fold_df[target_column].value_counts().to_dict()
        for label in labels:
            row[f"class_{label}_rows"] = int(counts.get(label, 0))
        diag_rows.append(row)

    signature_df = (
        df.assign(signature=df.groupby("score_pid")[target_column].transform(str))
        .groupby("score_pid")
        .agg(builds=("build", lambda s: "|".join(s.astype(str))), signature=(target_column, lambda s: "|".join(s.astype(str))))
        .reset_index()
        .merge(assignment_df[["score_pid", "fold_id"]], on="score_pid", how="left")
    )

    return pd.DataFrame(diag_rows).sort_values("fold_id").reset_index(drop=True), signature_df


def mean_std_prefix(values: list[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=0)),
    }


def stable_seed(*parts: object, offset: int = 0) -> int:
    key = "|".join(str(part) for part in parts)
    return int((RANDOM_STATE + offset + crc32(key.encode("utf-8"))) % (2**31 - 1))


def fit_and_predict_candidate_on_split(
    *,
    task: TaskSpec,
    pooled_df: pd.DataFrame,
    train_mask: pd.Series,
    calib_mask: pd.Series,
    test_mask: pd.Series,
    model_name: str,
    model_params: dict[str, Any],
    preprocess_name: str,
    imbalance_strategy: str,
    threshold_strategy: str,
    split_id: int,
    feature_config: FeatureReductionConfig,
) -> dict[str, Any]:
    X = pooled_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pooled_df[task.target_column].astype(int)

    X_train_all = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_cal_all = X.loc[calib_mask].reset_index(drop=True)
    y_cal = y.loc[calib_mask].reset_index(drop=True)
    X_train_final_all = X.loc[train_mask | calib_mask].reset_index(drop=True)
    y_train_final = y.loc[train_mask | calib_mask].reset_index(drop=True)
    X_test_all = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    calibration_features, calibration_selection_df = reduce_features_by_pearson(
        X_train_all,
        y_train,
        task=task,
        config=feature_config,
        split_id=split_id,
        selection_stage="calibration_train",
    )
    final_features, final_selection_df = reduce_features_by_pearson(
        X_train_final_all,
        y_train_final,
        task=task,
        config=feature_config,
        split_id=split_id,
        selection_stage="final_train_calibration",
    )

    X_train = X_train_all.loc[:, calibration_features]
    X_cal = X_cal_all.loc[:, calibration_features]
    X_train_final = X_train_final_all.loc[:, final_features]
    X_test = X_test_all.loc[:, final_features]

    train_seed = stable_seed(
        task.name,
        model_name,
        preprocess_name,
        imbalance_strategy,
        "calibration_fit",
        split_id,
    )
    final_seed = stable_seed(
        task.name,
        model_name,
        preprocess_name,
        imbalance_strategy,
        "final_fit",
        split_id,
        offset=5000,
    )

    calibration_model = fit_pipeline(
        task=task,
        model_name=model_name,
        model_params=model_params,
        preprocess_name=preprocess_name,
        imbalance_strategy=imbalance_strategy,
        X_train=X_train,
        y_train=y_train,
        random_state=train_seed,
    )

    if task.classification_type == "binary":
        cal_scores, score_source, default_threshold = score_binary_model(calibration_model, X_cal)
        numeric_threshold = select_threshold(y_cal.to_numpy(), cal_scores, threshold_strategy, default_threshold)
        y_cal_pred = threshold_predictions(cal_scores, numeric_threshold)
        calibration_metrics = binary_metrics(y_cal.to_numpy(), y_cal_pred)

        final_model = fit_pipeline(
            task=task,
            model_name=model_name,
            model_params=model_params,
            preprocess_name=preprocess_name,
            imbalance_strategy=imbalance_strategy,
            X_train=X_train_final,
            y_train=y_train_final,
            random_state=final_seed,
        )
        test_scores, _, _ = score_binary_model(final_model, X_test)
        y_test_pred = threshold_predictions(test_scores, numeric_threshold)
        test_metrics = binary_metrics(y_test.to_numpy(), y_test_pred)
        prediction_df = pooled_df.loc[test_mask, ["row_id", "score_pid", "build", task.target_column]].copy()
        prediction_df["y_pred"] = y_test_pred
        prediction_df["score_positive"] = test_scores
        prediction_df["split_id"] = split_id
        prediction_df["numeric_threshold"] = numeric_threshold
        prediction_df["n_features"] = len(final_features)

        return {
            "calibration_metrics": calibration_metrics,
            "test_metrics": test_metrics,
            "prediction_df": prediction_df,
            "numeric_threshold": float(numeric_threshold),
            "score_source": score_source,
            "selected_features": calibration_features,
            "final_selected_features": final_features,
            "feature_selection_df": pd.concat([calibration_selection_df, final_selection_df], ignore_index=True),
        }

    calibration_pred = np.asarray(calibration_model.predict(X_cal), dtype=int).reshape(-1)
    calibration_metrics = multiclass_metrics(y_cal.to_numpy(), calibration_pred)
    final_model = fit_pipeline(
        task=task,
        model_name=model_name,
        model_params=model_params,
        preprocess_name=preprocess_name,
        imbalance_strategy=imbalance_strategy,
        X_train=X_train_final,
        y_train=y_train_final,
        random_state=final_seed,
    )
    test_pred = np.asarray(final_model.predict(X_test), dtype=int).reshape(-1)
    test_metrics = multiclass_metrics(y_test.to_numpy(), test_pred)
    prediction_df = pooled_df.loc[test_mask, ["row_id", "score_pid", "build", task.target_column]].copy()
    prediction_df["y_pred"] = test_pred
    prediction_df["split_id"] = split_id
    prediction_df["n_features"] = len(final_features)

    return {
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "prediction_df": prediction_df,
        "numeric_threshold": None,
        "score_source": "multiclass_predict",
        "selected_features": calibration_features,
        "final_selected_features": final_features,
        "feature_selection_df": pd.concat([calibration_selection_df, final_selection_df], ignore_index=True),
    }


def fit_binary_calibration_param_on_split(
    *,
    task: TaskSpec,
    pooled_df: pd.DataFrame,
    train_mask: pd.Series,
    calib_mask: pd.Series,
    model_name: str,
    model_params: dict[str, Any],
    preprocess_name: str,
    imbalance_strategy: str,
    split_id: int,
    feature_config: FeatureReductionConfig,
) -> dict[str, Any]:
    X = pooled_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pooled_df[task.target_column].astype(int)

    X_train_all = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_cal_all = X.loc[calib_mask].reset_index(drop=True)
    y_cal = y.loc[calib_mask].reset_index(drop=True)

    selected_features, selection_df = reduce_features_by_pearson(
        X_train_all,
        y_train,
        task=task,
        config=feature_config,
        split_id=split_id,
        selection_stage="calibration_train",
    )
    X_train = X_train_all.loc[:, selected_features]
    X_cal = X_cal_all.loc[:, selected_features]

    train_seed = stable_seed(
        task.name,
        model_name,
        preprocess_name,
        imbalance_strategy,
        "calibration_fit",
        split_id,
    )
    calibration_model = fit_pipeline(
        task=task,
        model_name=model_name,
        model_params=model_params,
        preprocess_name=preprocess_name,
        imbalance_strategy=imbalance_strategy,
        X_train=X_train,
        y_train=y_train,
        random_state=train_seed,
    )
    cal_scores, score_source, default_threshold = score_binary_model(calibration_model, X_cal)

    threshold_outputs: dict[str, dict[str, Any]] = {}
    for threshold_strategy in task.threshold_strategies:
        numeric_threshold = select_threshold(
            y_cal.to_numpy(),
            cal_scores,
            threshold_strategy,
            default_threshold,
        )
        y_cal_pred = threshold_predictions(cal_scores, numeric_threshold)
        threshold_outputs[threshold_strategy] = {
            "numeric_threshold": float(numeric_threshold),
            "metrics": binary_metrics(y_cal.to_numpy(), y_cal_pred),
        }

    return {
        "score_source": score_source,
        "threshold_outputs": threshold_outputs,
        "selected_features": selected_features,
        "feature_selection_df": selection_df,
    }


def evaluate_binary_selected_param_on_split(
    *,
    task: TaskSpec,
    pooled_df: pd.DataFrame,
    train_mask: pd.Series,
    calib_mask: pd.Series,
    test_mask: pd.Series,
    model_name: str,
    model_params: dict[str, Any],
    preprocess_name: str,
    imbalance_strategy: str,
    threshold_strategy: str,
    split_id: int,
    feature_config: FeatureReductionConfig,
) -> dict[str, Any]:
    X = pooled_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pooled_df[task.target_column].astype(int)

    X_train_all = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_cal_all = X.loc[calib_mask].reset_index(drop=True)
    y_cal = y.loc[calib_mask].reset_index(drop=True)
    X_train_final_all = X.loc[train_mask | calib_mask].reset_index(drop=True)
    y_train_final = y.loc[train_mask | calib_mask].reset_index(drop=True)
    X_test_all = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    calibration_features, calibration_selection_df = reduce_features_by_pearson(
        X_train_all,
        y_train,
        task=task,
        config=feature_config,
        split_id=split_id,
        selection_stage="calibration_train",
    )
    final_features, final_selection_df = reduce_features_by_pearson(
        X_train_final_all,
        y_train_final,
        task=task,
        config=feature_config,
        split_id=split_id,
        selection_stage="final_train_calibration",
    )

    X_train = X_train_all.loc[:, calibration_features]
    X_cal = X_cal_all.loc[:, calibration_features]
    X_train_final = X_train_final_all.loc[:, final_features]
    X_test = X_test_all.loc[:, final_features]

    train_seed = stable_seed(
        task.name,
        model_name,
        preprocess_name,
        imbalance_strategy,
        "calibration_fit",
        split_id,
    )
    final_seed = stable_seed(
        task.name,
        model_name,
        preprocess_name,
        imbalance_strategy,
        "final_fit",
        split_id,
        offset=5000,
    )

    calibration_model = fit_pipeline(
        task=task,
        model_name=model_name,
        model_params=model_params,
        preprocess_name=preprocess_name,
        imbalance_strategy=imbalance_strategy,
        X_train=X_train,
        y_train=y_train,
        random_state=train_seed,
    )
    cal_scores, score_source, default_threshold = score_binary_model(calibration_model, X_cal)
    numeric_threshold = select_threshold(
        y_cal.to_numpy(),
        cal_scores,
        threshold_strategy,
        default_threshold,
    )
    y_cal_pred = threshold_predictions(cal_scores, numeric_threshold)
    calibration_metrics = binary_metrics(y_cal.to_numpy(), y_cal_pred)

    final_model = fit_pipeline(
        task=task,
        model_name=model_name,
        model_params=model_params,
        preprocess_name=preprocess_name,
        imbalance_strategy=imbalance_strategy,
        X_train=X_train_final,
        y_train=y_train_final,
        random_state=final_seed,
    )
    test_scores, _, _ = score_binary_model(final_model, X_test)
    y_test_pred = threshold_predictions(test_scores, numeric_threshold)
    test_metrics = binary_metrics(y_test.to_numpy(), y_test_pred)
    prediction_df = pooled_df.loc[test_mask, ["row_id", "score_pid", "build", task.target_column]].copy()
    prediction_df["y_pred"] = y_test_pred
    prediction_df["score_positive"] = test_scores
    prediction_df["split_id"] = split_id
    prediction_df["numeric_threshold"] = numeric_threshold
    prediction_df["n_features"] = len(final_features)

    return {
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "prediction_df": prediction_df,
        "numeric_threshold": float(numeric_threshold),
        "score_source": score_source,
        "selected_features": calibration_features,
        "final_selected_features": final_features,
        "feature_selection_df": pd.concat([calibration_selection_df, final_selection_df], ignore_index=True),
    }


def aggregate_candidate_results(
    *,
    task: TaskSpec,
    family_row: dict[str, Any],
    best_hyperparams: dict[str, Any],
    split_outputs: list[dict[str, Any]],
    feature_config: FeatureReductionConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    calibration_rows = [item["calibration_metrics"] for item in split_outputs]
    test_rows = [item["test_metrics"] for item in split_outputs]
    prediction_df = pd.concat([item["prediction_df"] for item in split_outputs], ignore_index=True)

    result: dict[str, Any] = {
        **family_row,
        "selected_hyperparams": json.dumps(best_hyperparams, sort_keys=True),
        **summarize_feature_selection(split_outputs, feature_config),
    }

    if task.classification_type == "binary":
        for metric in ["f1_positive", "f1_macro", "balanced_accuracy", "accuracy", "precision_positive", "recall_positive"]:
            result.update(mean_std_prefix([row[metric] for row in calibration_rows], f"calibration_{metric}"))
            result.update(mean_std_prefix([row[metric] for row in test_rows], f"test_{metric}"))

        thresholds = [item["numeric_threshold"] for item in split_outputs]
        result["selected_threshold_mean"] = float(np.mean(thresholds))
        result["selected_threshold_std"] = float(np.std(thresholds, ddof=0))
        result["score_source"] = split_outputs[0]["score_source"]

        overall = binary_metrics(
            prediction_df[task.target_column].to_numpy(dtype=int),
            prediction_df["y_pred"].to_numpy(dtype=int),
        )
        result["test_overall_f1_positive"] = float(overall["f1_positive"])
        result["test_overall_f1_macro"] = float(overall["f1_macro"])
        result["test_overall_balanced_accuracy"] = float(overall["balanced_accuracy"])
        result["test_overall_accuracy"] = float(overall["accuracy"])
        result["test_overall_tn"] = int(overall["tn"])
        result["test_overall_fp"] = int(overall["fp"])
        result["test_overall_fn"] = int(overall["fn"])
        result["test_overall_tp"] = int(overall["tp"])
        return result, prediction_df

    for metric in ["macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"]:
        result.update(mean_std_prefix([row[metric] for row in calibration_rows], f"calibration_{metric}"))
        result.update(mean_std_prefix([row[metric] for row in test_rows], f"test_{metric}"))
    result["threshold_strategy"] = "N/A"
    result["selected_threshold_mean"] = "N/A"
    result["selected_threshold_std"] = "N/A"
    overall = multiclass_metrics(
        prediction_df[task.target_column].to_numpy(dtype=int),
        prediction_df["y_pred"].to_numpy(dtype=int),
    )
    result["test_overall_macro_f1"] = float(overall["macro_f1"])
    result["test_overall_weighted_f1"] = float(overall["weighted_f1"])
    result["test_overall_balanced_accuracy"] = float(overall["balanced_accuracy"])
    result["test_overall_accuracy"] = float(overall["accuracy"])
    result["test_overall_confusion_matrix_json"] = overall["confusion_matrix_json"]
    return result, prediction_df


def candidate_family_rows(task: TaskSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = available_models()
    imbalance_map = BINARY_IMBALANCE_BY_MODEL if task.classification_type == "binary" else MULTICLASS_IMBALANCE_BY_MODEL
    for model_name in models:
        for preprocess_name in PREPROCESSING_CANDIDATES[model_name]:
            for imbalance_strategy in imbalance_map[model_name]:
                rows.append(
                    {
                        "model_name": model_name,
                        "preprocess_strategy": preprocess_name,
                        "imbalance_strategy": imbalance_strategy,
                    }
                )
    return rows


def choose_best_hyperparams_for_family(
    *,
    task: TaskSpec,
    pooled_df: pd.DataFrame,
    split_masks: list[tuple[pd.Series, pd.Series, pd.Series]],
    family_row: dict[str, Any],
    feature_config: FeatureReductionConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if task.classification_type == "binary":
        records: list[dict[str, Any]] = []
        best_params_by_threshold: dict[str, dict[str, Any]] = {}
        for param_idx, model_params in enumerate(MODEL_GRIDS[family_row["model_name"]]):
            per_split_outputs: list[dict[str, Any]] = []
            for split_id, (train_mask, calib_mask, _test_mask) in enumerate(split_masks):
                output = fit_binary_calibration_param_on_split(
                    task=task,
                    pooled_df=pooled_df,
                    train_mask=train_mask,
                    calib_mask=calib_mask,
                    model_name=family_row["model_name"],
                    model_params=model_params,
                    preprocess_name=family_row["preprocess_strategy"],
                    imbalance_strategy=family_row["imbalance_strategy"],
                    split_id=split_id,
                    feature_config=feature_config,
                )
                per_split_outputs.append(output)

            for threshold_strategy in task.threshold_strategies:
                calibration_metrics = [
                    item["threshold_outputs"][threshold_strategy]["metrics"]
                    for item in per_split_outputs
                ]
                row = {
                    **family_row,
                    "threshold_strategy": threshold_strategy,
                    "param_idx": param_idx,
                    "model_params": json.dumps(model_params, sort_keys=True),
                    "score_source": per_split_outputs[0]["score_source"],
                    **summarize_feature_selection(per_split_outputs, feature_config),
                }
                for metric in ["f1_positive", "f1_macro", "balanced_accuracy", "accuracy", "precision_positive", "recall_positive"]:
                    row.update(mean_std_prefix([m[metric] for m in calibration_metrics], metric))
                records.append(row)

        calib_df = pd.DataFrame(records)
        calib_df = calib_df.sort_values(
            by=[
                "threshold_strategy",
                "f1_positive_mean",
                "balanced_accuracy_mean",
                "precision_positive_mean",
                "recall_positive_mean",
                "param_idx",
            ],
            ascending=[True, False, False, False, False, True],
        ).reset_index(drop=True)

        for threshold_strategy in task.threshold_strategies:
            threshold_df = calib_df.loc[calib_df["threshold_strategy"].eq(threshold_strategy)].copy()
            best_params_by_threshold[threshold_strategy] = json.loads(threshold_df.iloc[0]["model_params"])
        return best_params_by_threshold, calib_df

    records: list[dict[str, Any]] = []
    for param_idx, model_params in enumerate(MODEL_GRIDS[family_row["model_name"]]):
        per_split_outputs: list[dict[str, Any]] = []
        for split_id, (train_mask, calib_mask, test_mask) in enumerate(split_masks):
            output = fit_and_predict_candidate_on_split(
                task=task,
                pooled_df=pooled_df,
                train_mask=train_mask,
                calib_mask=calib_mask,
                test_mask=test_mask,
                model_name=family_row["model_name"],
                model_params=model_params,
                preprocess_name=family_row["preprocess_strategy"],
                imbalance_strategy=family_row["imbalance_strategy"],
                threshold_strategy="N/A",
                split_id=split_id,
                feature_config=feature_config,
            )
            per_split_outputs.append(output)

        calibration_metrics = [item["calibration_metrics"] for item in per_split_outputs]
        row = {
            **family_row,
            "param_idx": param_idx,
            "model_params": json.dumps(model_params, sort_keys=True),
            **summarize_feature_selection(per_split_outputs, feature_config),
        }
        if task.classification_type == "binary":
            for metric in ["f1_positive", "f1_macro", "balanced_accuracy", "accuracy", "precision_positive", "recall_positive"]:
                row.update(mean_std_prefix([m[metric] for m in calibration_metrics], metric))
        else:
            for metric in ["macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"]:
                row.update(mean_std_prefix([m[metric] for m in calibration_metrics], metric))
        records.append(row)

    calib_df = pd.DataFrame(records)
    if task.classification_type == "binary":
        calib_df = calib_df.sort_values(
            by=["f1_positive_mean", "balanced_accuracy_mean", "precision_positive_mean", "recall_positive_mean", "param_idx"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
    else:
        calib_df = calib_df.sort_values(
            by=["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean", "param_idx"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    best_params = json.loads(calib_df.loc[0, "model_params"])
    return best_params, calib_df


def build_split_masks(pooled_df: pd.DataFrame, assignment_df: pd.DataFrame) -> list[tuple[pd.Series, pd.Series, pd.Series]]:
    merged = pooled_df[["score_pid"]].merge(assignment_df[["score_pid", "fold_id"]], on="score_pid", how="left")
    split_masks: list[tuple[pd.Series, pd.Series, pd.Series]] = []
    for split_id in range(N_FOLDS):
        test_fold = split_id
        calib_fold = (split_id + 1) % N_FOLDS
        test_mask = merged["fold_id"].eq(test_fold)
        calib_mask = merged["fold_id"].eq(calib_fold)
        train_mask = ~(test_mask | calib_mask)
        split_masks.append((train_mask, calib_mask, test_mask))
    return split_masks


def run_task(task: TaskSpec, feature_config: FeatureReductionConfig) -> dict[str, pd.DataFrame]:
    print("=" * 120)
    print(f"Running task: {task.name}")
    print("=" * 120)

    task_dir = OUTPUT_DIR / task.name
    task_dir.mkdir(parents=True, exist_ok=True)

    pooled_df = load_pooled_task_dataframe(task)
    assignment_df = assign_grouped_stratified_folds(pooled_df, task.target_column, N_FOLDS, RANDOM_STATE)
    fold_diag_df, signature_df = fold_diagnostics(pooled_df, assignment_df, task.target_column)
    split_masks = build_split_masks(pooled_df, assignment_df)

    assignment_df.to_csv(task_dir / f"{task.name}_participant_fold_assignment.csv", index=False)
    fold_diag_df.to_csv(task_dir / f"{task.name}_fold_diagnostics.csv", index=False)
    signature_df.to_csv(task_dir / f"{task.name}_participant_signatures.csv", index=False)

    family_rows = candidate_family_rows(task)
    calibration_records: list[pd.DataFrame] = []
    final_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    feature_selection_tables: list[pd.DataFrame] = []

    for family_idx, family_row in enumerate(family_rows, start=1):
        print(
            f"[{task.name}] family {family_idx}/{len(family_rows)} | "
            f"{family_row['model_name']} | {family_row['preprocess_strategy']} | "
            f"{family_row['imbalance_strategy']}"
        )

        best_hyperparams, calib_df = choose_best_hyperparams_for_family(
            task=task,
            pooled_df=pooled_df,
            split_masks=split_masks,
            family_row=family_row,
            feature_config=feature_config,
        )
        calibration_records.append(calib_df)

        if task.classification_type == "binary":
            assert isinstance(best_hyperparams, dict)
            for threshold_strategy, selected_params in best_hyperparams.items():
                split_outputs: list[dict[str, Any]] = []
                for split_id, (train_mask, calib_mask, test_mask) in enumerate(split_masks):
                    split_outputs.append(
                        evaluate_binary_selected_param_on_split(
                            task=task,
                            pooled_df=pooled_df,
                            train_mask=train_mask,
                            calib_mask=calib_mask,
                            test_mask=test_mask,
                            model_name=family_row["model_name"],
                            model_params=selected_params,
                            preprocess_name=family_row["preprocess_strategy"],
                            imbalance_strategy=family_row["imbalance_strategy"],
                            threshold_strategy=threshold_strategy,
                            split_id=split_id,
                            feature_config=feature_config,
                        )
                    )
                feature_selection_tables.extend(
                    output["feature_selection_df"].assign(
                        model_name=family_row["model_name"],
                        preprocess_strategy=family_row["preprocess_strategy"],
                        imbalance_strategy=family_row["imbalance_strategy"],
                        threshold_strategy=threshold_strategy,
                    )
                    for output in split_outputs
                    if "feature_selection_df" in output
                )

                family_threshold_row = {
                    **family_row,
                    "threshold_strategy": threshold_strategy,
                }
                final_row, prediction_df = aggregate_candidate_results(
                    task=task,
                    family_row=family_threshold_row,
                    best_hyperparams=selected_params,
                    split_outputs=split_outputs,
                    feature_config=feature_config,
                )
                final_rows.append(final_row)
                prediction_df["model_name"] = family_row["model_name"]
                prediction_df["preprocess_strategy"] = family_row["preprocess_strategy"]
                prediction_df["imbalance_strategy"] = family_row["imbalance_strategy"]
                prediction_df["threshold_strategy"] = threshold_strategy
                prediction_tables.append(prediction_df)
        else:
            split_outputs = []
            for split_id, (train_mask, calib_mask, test_mask) in enumerate(split_masks):
                split_outputs.append(
                    fit_and_predict_candidate_on_split(
                        task=task,
                        pooled_df=pooled_df,
                        train_mask=train_mask,
                        calib_mask=calib_mask,
                        test_mask=test_mask,
                        model_name=family_row["model_name"],
                        model_params=best_hyperparams,
                        preprocess_name=family_row["preprocess_strategy"],
                        imbalance_strategy=family_row["imbalance_strategy"],
                        threshold_strategy="N/A",
                        split_id=split_id,
                        feature_config=feature_config,
                    )
                )
            feature_selection_tables.extend(
                output["feature_selection_df"].assign(
                    model_name=family_row["model_name"],
                    preprocess_strategy=family_row["preprocess_strategy"],
                    imbalance_strategy=family_row["imbalance_strategy"],
                    threshold_strategy="N/A",
                )
                for output in split_outputs
                if "feature_selection_df" in output
            )

            family_threshold_row = {
                **family_row,
                "threshold_strategy": "N/A",
            }
            final_row, prediction_df = aggregate_candidate_results(
                task=task,
                family_row=family_threshold_row,
                best_hyperparams=best_hyperparams,
                split_outputs=split_outputs,
                feature_config=feature_config,
            )
            final_rows.append(final_row)
            prediction_df["model_name"] = family_row["model_name"]
            prediction_df["preprocess_strategy"] = family_row["preprocess_strategy"]
            prediction_df["imbalance_strategy"] = family_row["imbalance_strategy"]
            prediction_df["threshold_strategy"] = "N/A"
            prediction_tables.append(prediction_df)

    calibration_df = pd.concat(calibration_records, ignore_index=True)
    final_df = pd.DataFrame(final_rows)

    if task.classification_type == "binary":
        final_df = final_df.sort_values(
            by=["test_f1_positive_mean", "test_balanced_accuracy_mean", "test_precision_positive_mean", "model_name"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    else:
        final_df = final_df.sort_values(
            by=["test_macro_f1_mean", "test_balanced_accuracy_mean", "test_accuracy_mean", "model_name"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    calibration_df.to_csv(task_dir / f"{task.name}_calibration_grid.csv", index=False)
    final_df.to_csv(task_dir / f"{task.name}_test_results.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(task_dir / f"{task.name}_test_predictions.csv", index=False)
    feature_selection_df = pd.concat(feature_selection_tables, ignore_index=True) if feature_selection_tables else pd.DataFrame()
    if not feature_selection_df.empty:
        task_level_cols = [
            "task",
            "split_id",
            "selection_stage",
            "feature_reduction_method",
            "corr_method",
            "corr_threshold",
            "group_id",
            "group_size",
            "selected_feature",
            "removed_features",
            "selected_target_abs_pearson",
            "group_features",
            "n_selected_features",
        ]
        feature_selection_df = (
            feature_selection_df.loc[:, [col for col in task_level_cols if col in feature_selection_df.columns]]
            .drop_duplicates()
            .sort_values(["task", "split_id", "selection_stage", "group_id"])
            .reset_index(drop=True)
        )
    feature_selection_df.to_csv(task_dir / f"{task.name}_feature_reduction_details.csv", index=False)

    return {
        "pooled_df": pooled_df,
        "assignment_df": assignment_df,
        "fold_diag_df": fold_diag_df,
        "calibration_df": calibration_df,
        "final_df": final_df,
        "feature_selection_df": feature_selection_df,
    }


def run_selected_task(task: TaskSpec, task_config: dict[str, Any], feature_config: FeatureReductionConfig) -> dict[str, pd.DataFrame]:
    print("=" * 100)
    print(f"Running selected article configuration: {task.name}")
    print("=" * 100)

    task_dir = OUTPUT_DIR / task.name
    task_dir.mkdir(parents=True, exist_ok=True)

    pooled_df = load_pooled_task_dataframe(task)
    assignment_df = assign_grouped_stratified_folds(pooled_df, task.target_column, N_FOLDS, RANDOM_STATE)
    fold_diag_df, signature_df = fold_diagnostics(pooled_df, assignment_df, task.target_column)
    split_masks = build_split_masks(pooled_df, assignment_df)

    assignment_df.to_csv(task_dir / f"{task.name}_participant_fold_assignment.csv", index=False)
    fold_diag_df.to_csv(task_dir / f"{task.name}_fold_diagnostics.csv", index=False)
    signature_df.to_csv(task_dir / f"{task.name}_participant_signatures.csv", index=False)

    model_name = task_config["model_name"]
    preprocess_name = task_config["preprocess_strategy"]
    imbalance_strategy = task_config["imbalance_strategy"]
    threshold_strategy = task_config.get("threshold_strategy", "N/A")
    selected_hyperparams = task_config["selected_hyperparams"]

    split_outputs: list[dict[str, Any]] = []
    for split_id, (train_mask, calib_mask, test_mask) in enumerate(split_masks):
        if task.classification_type == "binary":
            split_outputs.append(
                evaluate_binary_selected_param_on_split(
                    task=task,
                    pooled_df=pooled_df,
                    train_mask=train_mask,
                    calib_mask=calib_mask,
                    test_mask=test_mask,
                    model_name=model_name,
                    model_params=selected_hyperparams,
                    preprocess_name=preprocess_name,
                    imbalance_strategy=imbalance_strategy,
                    threshold_strategy=threshold_strategy,
                    split_id=split_id,
                    feature_config=feature_config,
                )
            )
        else:
            split_outputs.append(
                fit_and_predict_candidate_on_split(
                    task=task,
                    pooled_df=pooled_df,
                    train_mask=train_mask,
                    calib_mask=calib_mask,
                    test_mask=test_mask,
                    model_name=model_name,
                    model_params=selected_hyperparams,
                    preprocess_name=preprocess_name,
                    imbalance_strategy=imbalance_strategy,
                    threshold_strategy="N/A",
                    split_id=split_id,
                    feature_config=feature_config,
                )
            )

    family_row = {
        "model_name": model_name,
        "preprocess_strategy": preprocess_name,
        "imbalance_strategy": imbalance_strategy,
        "threshold_strategy": threshold_strategy if task.classification_type == "binary" else "N/A",
    }
    final_row, prediction_df = aggregate_candidate_results(
        task=task,
        family_row=family_row,
        best_hyperparams=selected_hyperparams,
        split_outputs=split_outputs,
        feature_config=feature_config,
    )

    feature_selection_tables = [
        output["feature_selection_df"].assign(**family_row)
        for output in split_outputs
        if "feature_selection_df" in output
    ]
    feature_selection_df = pd.concat(feature_selection_tables, ignore_index=True) if feature_selection_tables else pd.DataFrame()
    final_df = pd.DataFrame([final_row])

    final_df.to_csv(task_dir / f"{task.name}_selected_results.csv", index=False)
    prediction_df.assign(**family_row).to_csv(task_dir / f"{task.name}_test_predictions.csv", index=False)
    feature_selection_df.to_csv(task_dir / f"{task.name}_feature_reduction_details.csv", index=False)

    return {
        "pooled_df": pooled_df,
        "assignment_df": assignment_df,
        "fold_diag_df": fold_diag_df,
        "final_df": final_df,
        "feature_selection_df": feature_selection_df,
    }


def load_saved_task_result(task: TaskSpec) -> dict[str, pd.DataFrame]:
    task_dir = OUTPUT_DIR / task.name
    result = {
        "fold_diag_df": pd.read_csv(task_dir / f"{task.name}_fold_diagnostics.csv"),
        "final_df": pd.read_csv(task_dir / f"{task.name}_test_results.csv"),
    }
    feature_selection_path = task_dir / f"{task.name}_feature_reduction_details.csv"
    if feature_selection_path.exists():
        result["feature_selection_df"] = pd.read_csv(feature_selection_path)
    return result


def short_result_table(df: pd.DataFrame, task: TaskSpec, top_n: int = 5) -> pd.DataFrame:
    if task.classification_type == "binary":
        cols = [
            "model_name",
            "preprocess_strategy",
            "imbalance_strategy",
            "threshold_strategy",
            "selected_hyperparams",
            "n_features_mean",
            "n_features_min",
            "n_features_max",
            "test_f1_positive_mean",
            "test_f1_macro_mean",
            "test_balanced_accuracy_mean",
            "test_accuracy_mean",
            "selected_threshold_mean",
        ]
    else:
        cols = [
            "model_name",
            "preprocess_strategy",
            "imbalance_strategy",
            "selected_hyperparams",
            "n_features_mean",
            "n_features_min",
            "n_features_max",
            "test_macro_f1_mean",
            "test_weighted_f1_mean",
            "test_balanced_accuracy_mean",
            "test_accuracy_mean",
        ]
    available = [column for column in cols if column in df.columns]
    return df.loc[:, available].head(top_n).copy()


def write_markdown_report(task_results: dict[str, dict[str, pd.DataFrame]], feature_config: FeatureReductionConfig) -> None:
    report_path = OUTPUT_DIR / "article_best_results.md"
    lines: list[str] = []
    lines.append("# Article Best-Configuration Pipeline Results")
    lines.append("")
    lines.append("## Methodological Summary")
    lines.append("")
    lines.append("This pipeline reproduces the selected article configuration for each questionnaire using pooled `Build A + Build B` data, 7 participant-grouped folds, a calibration fold, and Pearson-based feature reduction.")
    lines.append("")
    lines.append("Key design decisions:")
    lines.append("")
    lines.append("- pooled `A+B` dataset for each task;")
    lines.append("- initial input variables: 46 head-motion features + `vr_system_ordinal`;")
    if feature_config.enabled:
        lines.append(f"- feature reduction: head-motion metrics with absolute Pearson correlation `>{feature_config.corr_threshold:.2f}` are grouped inside each split, and the representative with the largest absolute Pearson correlation to the training target is kept;")
        lines.append("- `vr_system_ordinal` is retained as participant-context information and is not removed by the Pearson metric filter;")
    else:
        lines.append("- feature reduction disabled: all 47 input variables are used;")
    lines.append("- folds grouped by `score_pid`, so the same participant never appears in train/calibration and test simultaneously;")
    lines.append("- each participant contributes one `Build A` row and one `Build B` row, so build balance is naturally preserved at the participant level;")
    lines.append("- 7 outer folds with rotational roles: `TEST[i] = fold[i]`, `CALIBRATION[i] = fold[(i+1) mod 7]`, `TRAIN[i] = remaining 5 folds`;")
    lines.append("- model, preprocessing, imbalance handling, threshold strategy, and hyperparameters are read from `configs/pipeline_article_best.json`;")
    lines.append("- these selected configurations are evaluated without re-running the full experimental search;")
    lines.append("- final evaluation is performed on the held-out test fold after refitting on `TRAIN + CALIBRATION`.")
    lines.append("")
    lines.append("Important limitation:")
    lines.append("")
    lines.append("- Exact stratification is **not mathematically possible** with full participant grouping because a participant can change class between `Build A` and `Build B`. Therefore, the fold assignment used an **approximate grouped stratification** that balances class counts greedily at the participant level.")
    lines.append("")

    lines.append("## Best Configuration Per Questionnaire")
    lines.append("")
    best_rows: list[dict[str, Any]] = []
    for task_name, result in task_results.items():
        task = TASKS[task_name]
        best = result["final_df"].iloc[0].to_dict()
        threshold_strategy = best.get("threshold_strategy", "N/A")
        if pd.isna(threshold_strategy) or str(threshold_strategy).strip() == "":
            threshold_strategy = "N/A"
        row: dict[str, Any] = {
            "task": task_name,
            "model_name": best["model_name"],
            "preprocess_strategy": best["preprocess_strategy"],
            "imbalance_strategy": best["imbalance_strategy"],
            "threshold_strategy": threshold_strategy,
            "selected_hyperparams": best["selected_hyperparams"],
            "n_features_mean": round(float(best.get("n_features_mean", np.nan)), 6),
            "n_features_min": best.get("n_features_min", np.nan),
            "n_features_max": best.get("n_features_max", np.nan),
        }
        if task.classification_type == "binary":
            row.update(
                {
                    "test_f1_positive_mean": round(float(best["test_f1_positive_mean"]), 6),
                    "test_f1_macro_mean": round(float(best["test_f1_macro_mean"]), 6),
                    "test_balanced_accuracy_mean": round(float(best["test_balanced_accuracy_mean"]), 6),
                    "test_accuracy_mean": round(float(best["test_accuracy_mean"]), 6),
                }
            )
        else:
            row.update(
                {
                    "test_macro_f1_mean": round(float(best["test_macro_f1_mean"]), 6),
                    "test_weighted_f1_mean": round(float(best["test_weighted_f1_mean"]), 6),
                    "test_balanced_accuracy_mean": round(float(best["test_balanced_accuracy_mean"]), 6),
                    "test_accuracy_mean": round(float(best["test_accuracy_mean"]), 6),
                }
            )
        best_rows.append(row)

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUTPUT_DIR / "article_best_summary.csv", index=False)
    lines.append(best_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Selected Result By Questionnaire")
    lines.append("")
    top_frames: list[pd.DataFrame] = []
    for task_name, result in task_results.items():
        task = TASKS[task_name]
        top_df = short_result_table(result["final_df"], task, top_n=1).copy()
        top_df.insert(0, "task", task_name)
        top_frames.append(top_df)
        lines.append(f"### {task_name}")
        lines.append("")
        lines.append(top_df.drop(columns=["task"]).to_markdown(index=False))
        lines.append("")

    if top_frames:
        pd.concat(top_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "article_best_selected_by_questionnaire.csv",
            index=False,
        )

    lines.append("## Fold Diagnostics")
    lines.append("")
    for task_name, result in task_results.items():
        lines.append(f"### {task_name}")
        lines.append("")
        lines.append(result["fold_diag_df"].to_markdown(index=False))
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")

    config = load_json_config(args.config)

    global HEADFEATURES_DIR, OUTPUT_DIR, TASKS, N_FOLDS, RANDOM_STATE
    HEADFEATURES_DIR = resolve_config_path(config.get("headfeatures_dir", "headfeatures_data"))
    OUTPUT_DIR = resolve_config_path(config.get("output_dir", "outputs/article_best"))
    N_FOLDS = int(config.get("split", {}).get("n_folds", N_FOLDS))
    RANDOM_STATE = int(config.get("random_state", RANDOM_STATE))
    TASKS = {name: task_with_headfeatures_dir(task, HEADFEATURES_DIR) for name, task in TASKS.items()}

    ensure_output_dir()
    feature_reduction_config = config.get("feature_reduction", {})
    feature_config = FeatureReductionConfig(
        enabled=bool(feature_reduction_config.get("enabled", True)),
        corr_method=str(feature_reduction_config.get("corr_method", FEATURE_REDUCTION_CORR_METHOD)),
        corr_threshold=float(feature_reduction_config.get("corr_threshold", FEATURE_REDUCTION_THRESHOLD)),
    )
    task_results: dict[str, dict[str, pd.DataFrame]] = {}
    selected_tasks = args.task if args.task else list(TASKS)
    configured_tasks = config.get("tasks", {})
    for task_name in selected_tasks:
        if task_name not in configured_tasks:
            raise KeyError(f"Task `{task_name}` is not configured in {args.config}.")
        task = TASKS[task_name]
        task_results[task_name] = run_selected_task(task, configured_tasks[task_name], feature_config)
    write_markdown_report(task_results, feature_config)


if __name__ == "__main__":
    main()
