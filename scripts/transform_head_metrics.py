from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from head_metrics_schema import FEATURE_COLUMNS, missing_required_columns


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}. Use .csv, .xlsx, .xls, or .parquet.")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path}. Use .csv, .xlsx, .xls, or .parquet.")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_target_to_source_mapping(config: dict[str, Any]) -> dict[str, str]:
    if "target_to_source_mapping" in config:
        return dict(config["target_to_source_mapping"])

    source_to_target = dict(config.get("source_to_target_mapping", {}))
    return {target: source for source, target in source_to_target.items()}


def standardize_head_metrics(
    df: pd.DataFrame,
    *,
    target_to_source_mapping: dict[str, str] | None = None,
    passthrough_columns: list[str] | None = None,
    target_column: str | None = None,
    fill_missing_features: bool = False,
    missing_value: float = 0.0,
) -> pd.DataFrame:
    target_to_source_mapping = target_to_source_mapping or {}
    passthrough_columns = passthrough_columns or []

    output = pd.DataFrame(index=df.index)

    for feature in FEATURE_COLUMNS:
        source = target_to_source_mapping.get(feature, feature)
        if source in df.columns:
            output[feature] = pd.to_numeric(df[source], errors="coerce")
        elif fill_missing_features:
            output[feature] = missing_value
        else:
            raise ValueError(
                f"Missing required feature '{feature}'. "
                f"Expected source column '{source}'. Add it to the dataset or mapping."
            )

    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column not found in input dataset: {target_column}")
        output[target_column] = df[target_column]

    for column in passthrough_columns:
        if column not in df.columns:
            raise ValueError(f"Passthrough column not found in input dataset: {column}")
        output[column] = df[column]

    return output


def validate_standardized_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    missing = missing_required_columns(set(df.columns))
    numeric_issues = []
    for feature in FEATURE_COLUMNS:
        if feature in df.columns and not pd.api.types.is_numeric_dtype(df[feature]):
            numeric_issues.append(feature)

    result: dict[str, Any] = {
        "rows": int(len(df)),
        "feature_count": len(FEATURE_COLUMNS) - len(missing),
        "expected_feature_count": len(FEATURE_COLUMNS),
        "missing_features": missing,
        "non_numeric_features": numeric_issues,
        "target_column": target_column,
        "target_present": bool(target_column and target_column in df.columns),
    }
    if target_column and target_column in df.columns:
        result["target_distribution"] = {
            str(k): int(v) for k, v in df[target_column].value_counts(dropna=False).sort_index().items()
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a user head-metrics dataset to the 47-feature FAST schema.")
    parser.add_argument("--input", required=True, help="Input .csv/.xlsx/.parquet file.")
    parser.add_argument("--output", required=True, help="Standardized output file.")
    parser.add_argument("--config", default=None, help="Optional JSON config with column mapping and target column.")
    parser.add_argument("--target-column", default=None, help="Target column to preserve.")
    parser.add_argument("--passthrough-columns", nargs="*", default=None, help="Metadata columns to preserve.")
    parser.add_argument("--fill-missing-features", action="store_true", help="Fill missing required features.")
    parser.add_argument("--missing-value", type=float, default=0.0, help="Fill value when --fill-missing-features is used.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config) if args.config else {}
    target_column = args.target_column or config.get("target_column")
    passthrough_columns = args.passthrough_columns
    if passthrough_columns is None:
        passthrough_columns = list(config.get("passthrough_columns", []))

    df = read_table(args.input)
    standardized = standardize_head_metrics(
        df,
        target_to_source_mapping=build_target_to_source_mapping(config),
        passthrough_columns=passthrough_columns,
        target_column=target_column,
        fill_missing_features=bool(args.fill_missing_features or config.get("fill_missing_features", False)),
        missing_value=float(config.get("missing_value", args.missing_value)),
    )
    validation = validate_standardized_dataset(standardized, target_column)
    if validation["missing_features"]:
        raise ValueError(f"Missing features after standardization: {validation['missing_features']}")
    if validation["non_numeric_features"]:
        raise ValueError(f"Non-numeric standardized features: {validation['non_numeric_features']}")

    write_table(standardized, args.output)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    print(f"Saved standardized dataset: {args.output}")


if __name__ == "__main__":
    main()
