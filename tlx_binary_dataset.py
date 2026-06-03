from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from fast_tracking_ssq_dataset import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parent
HEADFEATURES_DIR = ROOT / "headfeatures_data"

TLX_RAW_MEAN_COL = "raw_tlx_mean_0_20"
TLX_RAW_SCORE_COL = "raw_tlx_score_0_100"
TLX_RAW_CLASS_COL = "raw_tlx_class"
TLX_BINARY_TARGET_COL = "tlx_not_low_target"
TLX_LOW_CLASS = "low"

SCORES_OUTPUT_PATH = ROOT / "tlx_binary_scores_and_classes.csv"
RAW_CLASS_DIST_OUTPUT_PATH = ROOT / "tlx_raw_class_distribution.csv"
BINARY_DIST_OUTPUT_PATH = ROOT / "tlx_binary_class_distribution.csv"
SUMMARY_OUTPUT_PATH = ROOT / "tlx_binary_distribution_summary.csv"

TLX_CLASSES = [
    "low",
    "low_to_moderate",
    "moderate_typical",
    "moderately_high",
    "high",
    "very_high",
]


def normalize_build(build: str) -> str:
    build = build.upper().strip()
    if build not in {"A", "B"}:
        raise ValueError(f"Build invalido: {build}. Use 'A' ou 'B'.")
    return build


def source_dataset_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSTLX_Build{build}.xlsx"


def source_metadata_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSTLX_Build{build}_metadata.csv"


def binary_dataset_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSTLXBinary_Build{build}.xlsx"


def binary_metadata_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSTLXBinary_Build{build}_metadata.csv"


def tlx_item_columns(build: str) -> list[str]:
    build = normalize_build(build)
    return [f"{build}-TLX{i}_1" for i in range(1, 7)]


def compute_raw_tlx_mean_0_20(df: pd.DataFrame, build: str) -> pd.Series:
    return df[tlx_item_columns(build)].mean(axis=1)


def scale_raw_tlx_to_100(raw_mean_0_20: pd.Series) -> pd.Series:
    return raw_mean_0_20 * 5.0


def classify_raw_tlx(score_0_100: float) -> str:
    if score_0_100 < 33:
        return "low"
    if score_0_100 < 41:
        return "low_to_moderate"
    if score_0_100 < 47:
        return "moderate_typical"
    if score_0_100 < 53:
        return "moderately_high"
    if score_0_100 <= 57:
        return "high"
    return "very_high"


def build_binary_target(raw_class: pd.Series) -> pd.Series:
    return raw_class.ne(TLX_LOW_CLASS).astype(int)


def summarize_multiclass_distribution(labels: pd.Series, build: str) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    counts = labels.value_counts().reindex(TLX_CLASSES, fill_value=0)
    proportions = counts / counts.sum()
    nonzero = proportions[proportions > 0]

    entropy = float(-(nonzero * nonzero.map(math.log)).sum()) if not nonzero.empty else 0.0
    normalized_entropy = entropy / math.log(len(TLX_CLASSES)) if len(TLX_CLASSES) > 1 else 0.0
    gini = float(1.0 - (proportions**2).sum())

    distribution_df = pd.DataFrame(
        {
            "build": build,
            "class_name": counts.index,
            "count": counts.values,
            "pct": (proportions.values * 100).round(4),
        }
    )

    summary = {
        "build": build,
        "largest_raw_class": str(counts.idxmax()),
        "largest_raw_class_pct": float(proportions.max() * 100),
        "raw_classes_used": int((counts > 0).sum()),
        "raw_normalized_entropy": float(normalized_entropy),
        "raw_gini": float(gini),
    }
    return distribution_df, summary


def summarize_binary_distribution(labels: pd.Series, build: str) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    counts = labels.value_counts().reindex([0, 1], fill_value=0)
    proportions = counts / counts.sum()

    distribution_df = pd.DataFrame(
        {
            "build": build,
            "tlx_not_low_target": counts.index,
            "count": counts.values,
            "pct": (proportions.values * 100).round(4),
        }
    )

    summary = {
        "build": build,
        "binary_negative_count": int(counts.loc[0]),
        "binary_negative_pct": float(proportions.loc[0] * 100),
        "binary_positive_count": int(counts.loc[1]),
        "binary_positive_pct": float(proportions.loc[1] * 100),
    }
    return distribution_df, summary


def build_tlx_binary_dataset(
    build: str,
    *,
    headfeatures_dir: str | Path = HEADFEATURES_DIR,
    overwrite: bool = False,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    build = normalize_build(build)
    dataset_path = binary_dataset_path(build, headfeatures_dir)
    metadata_path = binary_metadata_path(build, headfeatures_dir)

    if dataset_path.exists() and metadata_path.exists() and not overwrite:
        df = pd.read_excel(dataset_path)
        metadata_df = pd.read_csv(metadata_path)
        return dataset_path, metadata_path, df, metadata_df

    source_df = pd.read_excel(source_dataset_path(build, headfeatures_dir))
    metadata_df = pd.read_csv(source_metadata_path(build, headfeatures_dir))

    if len(source_df) != len(metadata_df):
        raise ValueError(
            f"Dataset e metadata do Build {build} possuem tamanhos diferentes: "
            f"{len(source_df)} vs {len(metadata_df)}."
        )

    raw_mean_0_20 = compute_raw_tlx_mean_0_20(source_df, build).round(4)
    raw_score_0_100 = scale_raw_tlx_to_100(raw_mean_0_20).round(4)
    raw_class = raw_score_0_100.apply(classify_raw_tlx)
    binary_target = build_binary_target(raw_class)

    output_df = source_df.copy()
    output_df[TLX_RAW_MEAN_COL] = raw_mean_0_20
    output_df[TLX_RAW_SCORE_COL] = raw_score_0_100
    output_df[TLX_RAW_CLASS_COL] = raw_class
    output_df[TLX_BINARY_TARGET_COL] = binary_target

    ordered_columns = FEATURE_COLUMNS + tlx_item_columns(build) + [
        TLX_RAW_MEAN_COL,
        TLX_RAW_SCORE_COL,
        TLX_RAW_CLASS_COL,
        TLX_BINARY_TARGET_COL,
    ]
    output_df = output_df.loc[:, ordered_columns]

    output_metadata_df = metadata_df.copy()
    output_metadata_df[TLX_RAW_MEAN_COL] = raw_mean_0_20.to_numpy()
    output_metadata_df[TLX_RAW_SCORE_COL] = raw_score_0_100.to_numpy()
    output_metadata_df[TLX_RAW_CLASS_COL] = raw_class.to_numpy()
    output_metadata_df[TLX_BINARY_TARGET_COL] = binary_target.to_numpy()

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(dataset_path, index=False)
    output_metadata_df.to_csv(metadata_path, index=False)

    return dataset_path, metadata_path, output_df, output_metadata_df


def build_distribution_exports(*, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_scores: list[pd.DataFrame] = []
    all_raw_dists: list[pd.DataFrame] = []
    all_binary_dists: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for build in ("A", "B"):
        _, metadata_path_b, df, meta = build_tlx_binary_dataset(build, headfeatures_dir=headfeatures_dir, overwrite=False)
        _ = metadata_path_b

        score_df = meta[["row_index", "score_pid", "matched_tracking_pid", "tracking_file"]].copy()
        score_df.insert(0, "build", build)
        for col in [TLX_RAW_MEAN_COL, TLX_RAW_SCORE_COL, TLX_RAW_CLASS_COL, TLX_BINARY_TARGET_COL]:
            score_df[col] = df[col].to_numpy()
        all_scores.append(score_df)

        raw_dist_df, raw_summary = summarize_multiclass_distribution(score_df[TLX_RAW_CLASS_COL], build)
        binary_dist_df, binary_summary = summarize_binary_distribution(score_df[TLX_BINARY_TARGET_COL], build)

        score_summary = {
            "build": build,
            "n_samples": int(len(score_df)),
            "mean_raw_tlx_0_100": float(score_df[TLX_RAW_SCORE_COL].mean()),
            "median_raw_tlx_0_100": float(score_df[TLX_RAW_SCORE_COL].median()),
            "std_raw_tlx_0_100": float(score_df[TLX_RAW_SCORE_COL].std()),
            "min_raw_tlx_0_100": float(score_df[TLX_RAW_SCORE_COL].min()),
            "q1_raw_tlx_0_100": float(score_df[TLX_RAW_SCORE_COL].quantile(0.25)),
            "q3_raw_tlx_0_100": float(score_df[TLX_RAW_SCORE_COL].quantile(0.75)),
            "max_raw_tlx_0_100": float(score_df[TLX_RAW_SCORE_COL].max()),
            **raw_summary,
            **binary_summary,
        }

        all_raw_dists.append(raw_dist_df)
        all_binary_dists.append(binary_dist_df)
        summary_rows.append(score_summary)

    scores_df = pd.concat(all_scores, ignore_index=True)
    raw_dist_df = pd.concat(all_raw_dists, ignore_index=True)
    binary_dist_df = pd.concat(all_binary_dists, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    scores_df.to_csv(SCORES_OUTPUT_PATH, index=False)
    raw_dist_df.to_csv(RAW_CLASS_DIST_OUTPUT_PATH, index=False)
    binary_dist_df.to_csv(BINARY_DIST_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    return scores_df, binary_dist_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera datasets binarios de TLX e exporta distribuicoes por build."
    )
    parser.add_argument(
        "--build",
        default="all",
        choices=("A", "B", "all", "a", "b"),
        help="Build a ser gerado. Use 'all' para A e B.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve os arquivos binarios caso ja existam.",
    )
    parser.add_argument(
        "--headfeatures-dir",
        default=str(HEADFEATURES_DIR),
        help="Diretorio com os datasets originais de TLX.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builds = ("A", "B") if str(args.build).lower() == "all" else (normalize_build(args.build),)

    for build in builds:
        dataset_path, metadata_path, df, _ = build_tlx_binary_dataset(
            build,
            headfeatures_dir=args.headfeatures_dir,
            overwrite=args.overwrite,
        )
        positive_count = int(df[TLX_BINARY_TARGET_COL].sum())
        print(
            f"Build {build}: {dataset_path.name} | rows={len(df)} | "
            f"not_low={positive_count} | not_low_pct={positive_count / len(df):.2%}"
        )
        print(f"Metadata: {metadata_path.name}")

    scores_df, binary_dist_df, summary_df = build_distribution_exports(headfeatures_dir=args.headfeatures_dir)
    _ = scores_df
    _ = binary_dist_df
    print()
    print(f"Saved: {SCORES_OUTPUT_PATH}")
    print(f"Saved: {RAW_CLASS_DIST_OUTPUT_PATH}")
    print(f"Saved: {BINARY_DIST_OUTPUT_PATH}")
    print(f"Saved: {SUMMARY_OUTPUT_PATH}")
    print()
    print(summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
