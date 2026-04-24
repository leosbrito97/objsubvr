from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

from head_metrics_schema import FEATURE_COLUMNS, FEATURE_FAMILIES
from transform_head_metrics import read_table

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss
except Exception:  # pragma: no cover - optional dependency
    SMOTE = None
    NearMiss = None

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception:  # pragma: no cover - optional dependency
    BalancedRandomForestClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


RANDOM_STATE = 42


@dataclass
class Candidate:
    model_name: str
    model_params: dict[str, Any]
    preprocess: str
    imbalance_strategy: str
    threshold_strategy: str

    @property
    def id(self) -> str:
        params = ",".join(f"{k}={v}" for k, v in sorted(self.model_params.items()))
        return (
            f"{self.model_name}|pre={self.preprocess}|imb={self.imbalance_strategy}|"
            f"thr={self.threshold_strategy}|{params}"
        )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def selected_features(config: dict[str, Any]) -> list[str]:
    explicit = config.get("features")
    if explicit:
        return list(explicit)

    include_families = config.get("feature_families")
    if include_families:
        features: list[str] = []
        for family in include_families:
            if family not in FEATURE_FAMILIES:
                raise ValueError(f"Unknown feature family: {family}. Options: {sorted(FEATURE_FAMILIES)}")
            features.extend(FEATURE_FAMILIES[family])
        return [feature for feature in FEATURE_COLUMNS if feature in set(features)]

    return FEATURE_COLUMNS.copy()


def validate_target(y: pd.Series, classification_type: str) -> pd.Series:
    y = y.dropna().astype(int)
    if classification_type == "binary":
        values = set(y.unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Binary target must contain only 0/1 values. Found: {sorted(values)}")
    return y


def class_weight_for_binary(y: pd.Series) -> dict[int, float]:
    counts = y.value_counts().sort_index()
    negative = int(counts.get(0, 0))
    positive = int(counts.get(1, 0))
    if negative == 0 or positive == 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: negative / positive}


def class_weight_balanced(y: pd.Series) -> dict[int, float]:
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    n_classes = int(len(counts))
    return {int(label): total / (n_classes * int(count)) for label, count in counts.items()}


def build_preprocessor(name: str) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if name == "none" or name == "median_only":
        return steps
    if name == "robust":
        return steps + [("scaler", RobustScaler())]
    if name == "standard":
        return steps + [("scaler", StandardScaler())]
    if name == "power_standard":
        return steps + [
            ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scaler", StandardScaler()),
        ]
    raise ValueError(f"Unsupported preprocessing strategy: {name}")


def build_model(model_name: str, params: dict[str, Any], y_reference: pd.Series, classification_type: str) -> object:
    params = dict(params)
    model_name = model_name.lower()

    if model_name == "logreg":
        solver = "liblinear" if classification_type == "binary" else "lbfgs"
        kwargs: dict[str, Any] = {}
        if "penalty" in params:
            kwargs["penalty"] = str(params["penalty"])
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver=str(params.get("solver", solver)),
            max_iter=int(params.get("max_iter", 3000)),
            class_weight=params.get("class_weight"),
            random_state=RANDOM_STATE,
            **kwargs,
        )

    if model_name == "svm":
        return SVC(
            C=float(params.get("C", 1.0)),
            kernel=str(params.get("kernel", "linear")),
            gamma=params.get("gamma", "scale"),
            probability=True,
            class_weight=params.get("class_weight"),
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if params.get("max_depth") is None else int(params.get("max_depth")),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight=params.get("class_weight"),
            random_state=RANDOM_STATE,
            n_jobs=int(params.get("n_jobs", 1)),
        )

    if model_name == "balanced_random_forest":
        if BalancedRandomForestClassifier is None:
            raise ImportError("imbalanced-learn is required for balanced_random_forest.")
        return BalancedRandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=None if params.get("max_depth") is None else int(params.get("max_depth")),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            sampling_strategy=params.get("sampling_strategy", "all"),
            replacement=bool(params.get("replacement", True)),
            bootstrap=bool(params.get("bootstrap", False)),
            random_state=RANDOM_STATE,
            n_jobs=int(params.get("n_jobs", 1)),
        )

    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is required for model_name='catboost'.")
        if classification_type == "binary":
            weights = class_weight_for_binary(y_reference)
            class_weights = [weights.get(0, 1.0), weights.get(1, 1.0)] if params.get("class_weights") == "balanced" else params.get("class_weights")
            return CatBoostClassifier(
                depth=int(params.get("depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.03)),
                iterations=int(params.get("iterations", 300)),
                loss_function="Logloss",
                verbose=False,
                allow_writing_files=False,
                random_seed=RANDOM_STATE,
                class_weights=class_weights,
            )
        weights = class_weight_balanced(y_reference)
        return CatBoostClassifier(
            depth=int(params.get("depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            iterations=int(params.get("iterations", 300)),
            loss_function="MultiClass",
            verbose=False,
            allow_writing_files=False,
            random_seed=RANDOM_STATE,
            class_weights=[weights.get(label, 1.0) for label in sorted(y_reference.unique())],
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is required for model_name='xgboost'.")
        objective = "binary:logistic" if classification_type == "binary" else "multi:softprob"
        extra = {"num_class": int(y_reference.nunique())} if classification_type != "binary" else {}
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            objective=objective,
            eval_metric="logloss" if classification_type == "binary" else "mlogloss",
            tree_method=params.get("tree_method", "hist"),
            random_state=RANDOM_STATE,
            n_jobs=int(params.get("n_jobs", 1)),
            **extra,
        )

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required for model_name='lightgbm'.")
        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            num_leaves=int(params.get("num_leaves", 15)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            objective="binary" if classification_type == "binary" else "multiclass",
            class_weight=params.get("class_weight"),
            random_state=RANDOM_STATE,
            n_jobs=int(params.get("n_jobs", 1)),
            verbosity=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def apply_class_weight_params(
    model_name: str,
    params: dict[str, Any],
    y_reference: pd.Series,
    classification_type: str,
    imbalance_strategy: str,
) -> dict[str, Any]:
    params = dict(params)
    if imbalance_strategy != "class_weight":
        return params
    if model_name == "catboost":
        params["class_weights"] = "balanced"
    elif model_name in {"logreg", "svm", "random_forest", "lightgbm"}:
        params["class_weight"] = "balanced"
    return params


def build_pipeline(candidate: Candidate, y_reference: pd.Series, classification_type: str) -> Pipeline:
    params = apply_class_weight_params(
        candidate.model_name,
        candidate.model_params,
        y_reference,
        classification_type,
        candidate.imbalance_strategy,
    )
    model = build_model(candidate.model_name, params, y_reference, classification_type)
    return Pipeline(build_preprocessor(candidate.preprocess) + [("model", model)])


def random_undersample(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    counts = y.value_counts()
    min_count = int(counts.min())
    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    for label in counts.index:
        idx = y.index[y.eq(label)].to_numpy()
        if len(idx) > min_count:
            idx = rng.choice(idx, size=min_count, replace=False)
        selected.extend(idx.tolist())
    rng.shuffle(selected)
    return X.loc[selected].reset_index(drop=True), y.loc[selected].reset_index(drop=True)


def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if strategy in {"none", "class_weight"}:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    if strategy == "random_undersample":
        return random_undersample(X, y, random_state)
    if strategy == "smote":
        if SMOTE is None:
            raise ImportError("imbalanced-learn is required for SMOTE.")
        min_count = int(y.value_counts().min())
        if min_count < 2:
            return X.reset_index(drop=True), y.reset_index(drop=True)
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


def positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if np.ndim(proba) == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def threshold_from_scores(y_true: np.ndarray, scores: np.ndarray, strategy: str) -> float:
    if strategy in {"default", "default_0_5", "model_predict"}:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5
    if strategy == "roc_gmean":
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        gmean = np.sqrt(tpr * (1.0 - fpr))
        return float(thresholds[int(np.nanargmax(gmean))])
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    if strategy == "pr_f1":
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
        return float(thresholds[int(np.nanargmax(f1))])
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
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def candidate_grid(config: dict[str, Any]) -> list[Candidate]:
    models = config["models"]
    preprocessors = config.get("preprocessing", ["median_only"])
    imbalance_strategies = config.get("imbalance_strategies", ["none"])
    threshold_strategies = config.get("threshold_strategies", ["default"])
    candidates: list[Candidate] = []
    for model_spec in models:
        model_name = model_spec["name"]
        param_grid = model_spec.get("param_grid", [{}])
        for params in param_grid:
            for preprocess in preprocessors:
                for imbalance in imbalance_strategies:
                    for threshold in threshold_strategies:
                        candidates.append(
                            Candidate(
                                model_name=model_name,
                                model_params=dict(params),
                                preprocess=str(preprocess),
                                imbalance_strategy=str(imbalance),
                                threshold_strategy=str(threshold),
                            )
                        )
    return candidates


def run_cv_for_candidate(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    classification_type: str,
    cv_splits: int,
    cv_repeats: int,
    selection_metric: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, Any]] = []
    oof_scores = np.full(len(y), np.nan, dtype=float) if classification_type == "binary" else None
    oof_true = np.full(len(y), np.nan, dtype=float) if classification_type == "binary" else None

    for split_id, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train_raw, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_raw, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        X_train, y_train = apply_resampling(
            X_train_raw,
            y_train_raw,
            candidate.imbalance_strategy,
            RANDOM_STATE + split_id,
        )
        model = build_pipeline(candidate, y_train_raw, classification_type)
        model.fit(X_train, y_train)

        if classification_type == "binary":
            scores = positive_scores(model, X_valid)
            threshold = 0.5 if candidate.threshold_strategy in {"default", "default_0_5", "model_predict"} else threshold_from_scores(
                y_valid.to_numpy(), scores, candidate.threshold_strategy
            )
            y_pred = (scores >= threshold).astype(int)
            fold_rows.append(binary_metrics(y_valid.to_numpy(), y_pred))
            oof_scores[valid_idx] = scores
            oof_true[valid_idx] = y_valid.to_numpy()
        else:
            y_pred = model.predict(X_valid)
            fold_rows.append(multiclass_metrics(y_valid.to_numpy(), y_pred))

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "candidate_id": candidate.id,
        "model": candidate.model_name,
        "model_params": json.dumps(candidate.model_params, sort_keys=True),
        "preprocess": candidate.preprocess,
        "imbalance_strategy": candidate.imbalance_strategy,
        "threshold_strategy": candidate.threshold_strategy,
    }
    for column in fold_df.columns:
        summary[f"cv_{column}_mean"] = float(fold_df[column].mean())
        summary[f"cv_{column}_std"] = float(fold_df[column].std(ddof=0))
    summary["selection_metric"] = selection_metric
    summary["selection_value"] = float(summary.get(f"cv_{selection_metric}_mean", -math.inf))
    return summary, oof_scores if classification_type == "binary" else None


def fit_final_and_evaluate(
    candidate: Candidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    classification_type: str,
    threshold: float | None,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    X_fit, y_fit = apply_resampling(X_train, y_train, candidate.imbalance_strategy, RANDOM_STATE + 9999)
    model = build_pipeline(candidate, y_train, classification_type)
    model.fit(X_fit, y_fit)

    if classification_type == "binary":
        scores = positive_scores(model, X_test)
        threshold = 0.5 if threshold is None else float(threshold)
        y_pred = (scores >= threshold).astype(int)
        metrics = binary_metrics(y_test.to_numpy(), y_pred)
        metrics["threshold"] = threshold
        predictions = pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred, "score_positive": scores})
    else:
        y_pred = model.predict(X_test)
        metrics = multiclass_metrics(y_test.to_numpy(), y_pred)
        metrics["threshold"] = "N/A"
        predictions = pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": y_pred})
    return model, metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configurable classification experiments for standardized head metrics.")
    parser.add_argument("--config", required=True, help="JSON config file.")
    args = parser.parse_args()
    config = load_config(args.config)

    output_dir = Path(config.get("output_dir", "notebook/outputs/classification_pipeline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_table(config["train_path"])
    test_df = read_table(config["test_path"])
    target_column = config["target_column"]
    classification_type = config.get("classification_type", "binary")
    features = selected_features(config)

    missing = [feature for feature in features if feature not in train_df.columns or feature not in test_df.columns]
    if missing:
        raise ValueError(f"Missing configured features in train/test datasets: {missing}")
    if target_column not in train_df.columns or target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' must exist in both train and test datasets.")

    X_train = train_df[features].copy()
    y_train = validate_target(train_df[target_column], classification_type).reset_index(drop=True)
    X_train = X_train.loc[y_train.index].reset_index(drop=True)
    X_test = test_df[features].copy()
    y_test = validate_target(test_df[target_column], classification_type).reset_index(drop=True)
    X_test = X_test.loc[y_test.index].reset_index(drop=True)

    selection_metric = config.get("selection_metric", "f1_positive" if classification_type == "binary" else "macro_f1")
    cv_splits = int(config.get("cv_splits", 3))
    cv_repeats = int(config.get("cv_repeats", 3))

    rows: list[dict[str, Any]] = []
    oof_scores_by_candidate: dict[str, np.ndarray] = {}
    for candidate in candidate_grid(config):
        summary, oof_scores = run_cv_for_candidate(
            candidate,
            X_train,
            y_train,
            classification_type=classification_type,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            selection_metric=selection_metric,
        )
        rows.append(summary)
        if oof_scores is not None:
            oof_scores_by_candidate[candidate.id] = oof_scores

    cv_results = pd.DataFrame(rows).sort_values(["selection_value", "candidate_id"], ascending=[False, True])
    best_row = cv_results.iloc[0]
    best_candidate = next(candidate for candidate in candidate_grid(config) if candidate.id == best_row["candidate_id"])

    selected_threshold: float | None = None
    if classification_type == "binary":
        oof_scores = oof_scores_by_candidate[best_candidate.id]
        valid = ~np.isnan(oof_scores)
        selected_threshold = threshold_from_scores(
            y_train.to_numpy()[valid],
            oof_scores[valid],
            best_candidate.threshold_strategy,
        )

    model, test_metrics, predictions = fit_final_and_evaluate(
        best_candidate,
        X_train,
        y_train,
        X_test,
        y_test,
        classification_type=classification_type,
        threshold=selected_threshold,
    )

    cv_results_path = output_dir / "cv_results.csv"
    predictions_path = output_dir / "test_predictions.csv"
    metrics_path = output_dir / "test_metrics.json"
    model_path = output_dir / "best_model.pkl"

    write_table(cv_results, cv_results_path)
    write_table(predictions, predictions_path)
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": features,
                "target_column": target_column,
                "candidate": best_candidate.__dict__,
                "threshold": selected_threshold,
                "classification_type": classification_type,
            },
            f,
        )

    payload = {
        "best_candidate": best_candidate.__dict__,
        "features": features,
        "cv_selection": best_row.to_dict(),
        "test_metrics": test_metrics,
        "paths": {
            "cv_results": str(cv_results_path),
            "test_predictions": str(predictions_path),
            "best_model": str(model_path),
        },
    }
    write_json(payload, metrics_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
