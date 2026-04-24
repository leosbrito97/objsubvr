from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_tracking_ssq_dataset import (  # noqa: E402
    QUESTIONNAIRE_TYPES,
    build_headfeatures_questionnaire_dataset,
)
from spes_binary_dataset import build_spes_binary_dataset  # noqa: E402
from ssq_3class_dataset import build_ssq_3class_dataset  # noqa: E402
from sus_binary_dataset import build_sus_binary_dataset  # noqa: E402
from tlx_binary_dataset import build_tlx_binary_dataset  # noqa: E402


DERIVED_DATASET_BUILDERS = {
    "sus_binary": build_sus_binary_dataset,
    "tlx_binary": build_tlx_binary_dataset,
    "spes_binary": build_spes_binary_dataset,
    "ssq_3class": build_ssq_3class_dataset,
}

DERIVED_TO_QUESTIONNAIRE = {
    "sus_binary": "SUS",
    "tlx_binary": "TLX",
    "spes_binary": "SPES",
    "ssq_3class": "SSQ",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_builds(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        builds = [str(item).upper().strip() for item in value]
    else:
        raw = str(value).strip().lower()
        if raw == "all":
            builds = ["A", "B"]
        else:
            builds = [str(value).upper().strip()]

    for build in builds:
        if build not in {"A", "B"}:
            raise ValueError(f"Invalid build: {build}. Use A, B, or all.")
    return builds


def questionnaire_dataset_stem(questionnaire_type: str, build: str) -> str:
    return f"HeadFeaturesVS{questionnaire_type.upper()}_Build{build.upper()}"


def generate_headfeatures_for_questionnaire(
    *,
    questionnaire_type: str,
    builds: list[str],
    scores_path: str | Path,
    tracking_dir: str | Path,
    output_dir: str | Path,
    n_turns_threshold_deg: float,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for build in builds:
        dataset_df, metadata_df, skipped = build_headfeatures_questionnaire_dataset(
            build=build,
            questionnaire_type=questionnaire_type,
            scores_path=scores_path,
            tracking_dir=tracking_dir,
            n_turns_threshold_deg=n_turns_threshold_deg,
        )

        stem = questionnaire_dataset_stem(questionnaire_type, build)
        dataset_path = output_dir / f"{stem}.xlsx"
        metadata_path = output_dir / f"{stem}_metadata.csv"
        dataset_df.to_excel(dataset_path, index=False)
        metadata_df.to_csv(metadata_path, index=False)

        rows.append(
            {
                "stage": "headfeatures",
                "build": build,
                "questionnaire_type": questionnaire_type,
                "dataset_file": dataset_path.name,
                "metadata_file": metadata_path.name,
                "rows": int(len(dataset_df)),
                "columns": int(len(dataset_df.columns)),
                "skipped_count": int(len(skipped)),
                "skipped_participants": ", ".join(skipped),
            }
        )

    return pd.DataFrame(rows)


def generate_derived_dataset(
    *,
    derived_dataset: str,
    builds: list[str],
    output_dir: str | Path,
    overwrite: bool = True,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = DERIVED_DATASET_BUILDERS[derived_dataset]
    rows: list[dict[str, Any]] = []

    for build in builds:
        dataset_path, metadata_path, df, metadata_df = builder(
            build,
            headfeatures_dir=output_dir,
            overwrite=overwrite,
        )
        rows.append(
            {
                "stage": "derived",
                "build": build,
                "questionnaire_type": DERIVED_TO_QUESTIONNAIRE[derived_dataset],
                "derived_dataset": derived_dataset,
                "dataset_file": dataset_path.name,
                "metadata_file": metadata_path.name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "metadata_rows": int(len(metadata_df)),
            }
        )

    return pd.DataFrame(rows)


def run_feature_engineering(config: dict[str, Any]) -> dict[str, Any]:
    questionnaire_type = str(config.get("questionnaire_type", "SUS")).upper().strip()
    if questionnaire_type not in QUESTIONNAIRE_TYPES:
        raise ValueError(f"Invalid questionnaire_type: {questionnaire_type}. Use one of {QUESTIONNAIRE_TYPES}.")

    builds = normalize_builds(config.get("builds", ["A", "B"]))
    output_dir = Path(config["output_dir"])
    scores_path = Path(config["scores_path"])
    tracking_dir = Path(config["tracking_dir"])
    n_turns_threshold_deg = float(config.get("n_turns_threshold_deg", 0.05))
    derived_dataset = config.get("derived_dataset")

    if derived_dataset:
        derived_dataset = str(derived_dataset).lower().strip()
        if derived_dataset not in DERIVED_DATASET_BUILDERS:
            raise ValueError(f"Invalid derived_dataset: {derived_dataset}.")
        expected_questionnaire = DERIVED_TO_QUESTIONNAIRE[derived_dataset]
        if questionnaire_type != expected_questionnaire:
            raise ValueError(
                f"derived_dataset={derived_dataset} expects questionnaire_type={expected_questionnaire}, "
                f"got {questionnaire_type}."
            )

    headfeatures_manifest = generate_headfeatures_for_questionnaire(
        questionnaire_type=questionnaire_type,
        builds=builds,
        scores_path=scores_path,
        tracking_dir=tracking_dir,
        output_dir=output_dir,
        n_turns_threshold_deg=n_turns_threshold_deg,
    )

    derived_manifest = pd.DataFrame()
    if derived_dataset:
        derived_manifest = generate_derived_dataset(
            derived_dataset=derived_dataset,
            builds=builds,
            output_dir=output_dir,
            overwrite=True,
        )

    manifest = pd.concat([headfeatures_manifest, derived_manifest], ignore_index=True)
    manifest_path = output_dir / "feature_engineering_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return {
        "questionnaire_type": questionnaire_type,
        "builds": builds,
        "scores_path": str(scores_path),
        "tracking_dir": str(tracking_dir),
        "output_dir": str(output_dir),
        "derived_dataset": derived_dataset,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HeadFeaturesVS datasets from raw FAST-style scores + tracking, with optional derived targets."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config.")
    parser.add_argument("--scores-path", default=None, help="Path to FAST Scores.xlsx.")
    parser.add_argument("--tracking-dir", default=None, help="Directory with tracking CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory where generated spreadsheets will be written.")
    parser.add_argument("--questionnaire-type", default=None, choices=QUESTIONNAIRE_TYPES, help="Questionnaire type.")
    parser.add_argument("--builds", default=None, help="A, B, or all.")
    parser.add_argument(
        "--derived-dataset",
        default=None,
        choices=["sus_binary", "tlx_binary", "spes_binary", "ssq_3class"],
        help="Optional derived dataset to create after head feature extraction.",
    )
    parser.add_argument(
        "--n-turns-threshold-deg",
        type=float,
        default=None,
        help="Yaw threshold in degrees used by head_n_turns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config) if args.config else {}

    if args.scores_path is not None:
        config["scores_path"] = args.scores_path
    if args.tracking_dir is not None:
        config["tracking_dir"] = args.tracking_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.questionnaire_type is not None:
        config["questionnaire_type"] = args.questionnaire_type
    if args.builds is not None:
        config["builds"] = args.builds
    if args.derived_dataset is not None:
        config["derived_dataset"] = args.derived_dataset
    if args.n_turns_threshold_deg is not None:
        config["n_turns_threshold_deg"] = args.n_turns_threshold_deg

    required = ["scores_path", "tracking_dir", "output_dir", "questionnaire_type"]
    missing = [key for key in required if key not in config or config[key] in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    result = run_feature_engineering(config)
    print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2, ensure_ascii=False))
    print(result["manifest"].to_string(index=False))


if __name__ == "__main__":
    main()
