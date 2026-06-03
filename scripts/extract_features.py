from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from objsubvr.fast_tracking_ssq_dataset import QUESTIONNAIRE_TYPES
from scripts.feature_engineering_pipeline import generate_derived_dataset, generate_headfeatures_for_questionnaire


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "feature_extraction_fast.json"

DERIVED_BY_QUESTIONNAIRE = {
    "SUS": "sus_binary",
    "TLX": "tlx_binary",
    "SPES": "spes_binary",
    "SSQ": "ssq_3class",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HMD pose features and questionnaire target tables.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Feature-extraction JSON config.")
    parser.add_argument("--scores-path", default=None, help="Optional override for FAST Scores.xlsx.")
    parser.add_argument("--tracking-dir", default=None, help="Optional override for the tracking CSV directory.")
    parser.add_argument("--output-dir", default=None, help="Optional override for generated feature tables.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    scores_path = resolve_path(args.scores_path or config["scores_path"])
    tracking_dir = resolve_path(args.tracking_dir or config["tracking_dir"])
    output_dir = resolve_path(args.output_dir or config.get("output_dir", "headfeatures_data"))
    questionnaires = config.get("questionnaires", QUESTIONNAIRE_TYPES)
    builds = config.get("builds", ["A", "B"])
    n_turns_threshold_deg = float(config.get("n_turns_threshold_deg", 0.05))
    create_derived_targets = bool(config.get("create_derived_targets", True))

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_frames: list[pd.DataFrame] = []
    for questionnaire_type in questionnaires:
        manifest_frames.append(
            generate_headfeatures_for_questionnaire(
                questionnaire_type=questionnaire_type,
                builds=builds,
                scores_path=scores_path,
                tracking_dir=tracking_dir,
                output_dir=output_dir,
                n_turns_threshold_deg=n_turns_threshold_deg,
            )
        )
        if create_derived_targets:
            manifest_frames.append(
                generate_derived_dataset(
                    derived_dataset=DERIVED_BY_QUESTIONNAIRE[questionnaire_type],
                    builds=builds,
                    output_dir=output_dir,
                    overwrite=True,
                )
            )

    manifest = pd.concat(manifest_frames, ignore_index=True)
    manifest_path = output_dir / "feature_extraction_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Feature tables written to: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
