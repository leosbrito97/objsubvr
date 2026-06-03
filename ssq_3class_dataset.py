from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from fast_tracking_ssq_dataset import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parent
HEADFEATURES_DIR = ROOT / "headfeatures_data"

SSQ_RAW_NAUSEA_COL = "ssq_raw_nausea_sum"
SSQ_RAW_OCULOMOTOR_COL = "ssq_raw_oculomotor_sum"
SSQ_RAW_DISORIENTATION_COL = "ssq_raw_disorientation_sum"
SSQ_NAUSEA_SCORE_COL = "ssq_nausea_score"
SSQ_OCULOMOTOR_SCORE_COL = "ssq_oculomotor_score"
SSQ_DISORIENTATION_SCORE_COL = "ssq_disorientation_score"
SSQ_TOTAL_SCORE_COL = "ssq_total_score"
SSQ_5LEVEL_CLASS_COL = "ssq_total_class_5level"
SSQ_3CLASS_LABEL_COL = "ssq_total_class_3level"
SSQ_3CLASS_TARGET_COL = "ssq_3class_target"

SCORES_OUTPUT_PATH = ROOT / "ssq_3class_scores_and_classes.csv"
FIVELEVEL_DIST_OUTPUT_PATH = ROOT / "ssq_5level_class_distribution.csv"
THREECLASS_DIST_OUTPUT_PATH = ROOT / "ssq_3class_distribution.csv"
SUMMARY_OUTPUT_PATH = ROOT / "ssq_3class_distribution_summary.csv"

FIVELEVEL_CLASSES = [
    "negligible",
    "minimum",
    "significant",
    "concerning",
    "very_problematic",
]
THREECLASS_CLASSES = [
    "negligible",
    "intermediate",
    "very_problematic",
]
THREECLASS_TO_ID = {label: idx for idx, label in enumerate(THREECLASS_CLASSES)}

NAUSEA_ITEMS = [1, 6, 7, 8, 9, 15, 16]
OCULOMOTOR_ITEMS = [1, 2, 3, 4, 5, 9, 11, 12, 13]
DISORIENTATION_ITEMS = [5, 8, 9, 10, 11, 12, 13, 14]

NAUSEA_FACTOR = 9.54
OCULOMOTOR_FACTOR = 7.58
DISORIENTATION_FACTOR = 13.92
TOTAL_FACTOR = 3.74


def normalize_build(build: str) -> str:
    build = build.upper().strip()
    if build not in {"A", "B"}:
        raise ValueError(f"Build invalido: {build}. Use 'A' ou 'B'.")
    return build


def source_dataset_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSSSQ_Build{build}.xlsx"


def source_metadata_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSSSQ_Build{build}_metadata.csv"


def threeclass_dataset_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSSSQ3Class_Build{build}.xlsx"


def threeclass_metadata_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSSSQ3Class_Build{build}_metadata.csv"


def ssq_item_columns(build: str) -> list[str]:
    build = normalize_build(build)
    return [f"{build}-SSQ{i}" for i in range(1, 17)]


def ssq_col(build: str, item: int) -> str:
    return f"{normalize_build(build)}-SSQ{item}"


def compute_raw_sum(df: pd.DataFrame, build: str, items: list[int]) -> pd.Series:
    cols = [ssq_col(build, item) for item in items]
    return df[cols].sum(axis=1)


def classify_total_ssq_fivelevel(score: float) -> str:
    if score < 5:
        return "negligible"
    if score < 10:
        return "minimum"
    if score < 15:
        return "significant"
    if score <= 20:
        return "concerning"
    return "very_problematic"


def collapse_ssq_to_three_classes(fivelevel_label: str) -> str:
    if fivelevel_label == "negligible":
        return "negligible"
    if fivelevel_label == "very_problematic":
        return "very_problematic"
    return "intermediate"


def summarize_distribution(
    labels: pd.Series,
    build: str,
    class_labels: list[str],
    label_col: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    counts = labels.value_counts().reindex(class_labels, fill_value=0)
    proportions = counts / counts.sum()
    nonzero = proportions[proportions > 0]

    entropy = float(-(nonzero * nonzero.map(math.log)).sum()) if not nonzero.empty else 0.0
    normalized_entropy = entropy / math.log(len(class_labels)) if len(class_labels) > 1 else 0.0
    gini = float(1.0 - (proportions**2).sum())

    distribution_df = pd.DataFrame(
        {
            "build": build,
            label_col: counts.index,
            "count": counts.values,
            "pct": (proportions.values * 100).round(4),
        }
    )

    summary = {
        "build": build,
        f"{label_col}_largest_class": str(counts.idxmax()),
        f"{label_col}_largest_class_pct": float(proportions.max() * 100),
        f"{label_col}_classes_used": int((counts > 0).sum()),
        f"{label_col}_normalized_entropy": float(normalized_entropy),
        f"{label_col}_gini": float(gini),
    }
    return distribution_df, summary


def build_ssq_3class_dataset(
    build: str,
    *,
    headfeatures_dir: str | Path = HEADFEATURES_DIR,
    overwrite: bool = False,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    build = normalize_build(build)
    dataset_path = threeclass_dataset_path(build, headfeatures_dir)
    metadata_path = threeclass_metadata_path(build, headfeatures_dir)

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

    raw_n = compute_raw_sum(source_df, build, NAUSEA_ITEMS)
    raw_o = compute_raw_sum(source_df, build, OCULOMOTOR_ITEMS)
    raw_d = compute_raw_sum(source_df, build, DISORIENTATION_ITEMS)

    nausea_score = (raw_n * NAUSEA_FACTOR).round(4)
    oculomotor_score = (raw_o * OCULOMOTOR_FACTOR).round(4)
    disorientation_score = (raw_d * DISORIENTATION_FACTOR).round(4)
    total_score = ((raw_n + raw_o + raw_d) * TOTAL_FACTOR).round(4)
    fivelevel_class = total_score.apply(classify_total_ssq_fivelevel)
    threeclass_label = fivelevel_class.apply(collapse_ssq_to_three_classes)
    threeclass_target = threeclass_label.map(THREECLASS_TO_ID).astype(int)

    output_df = source_df.copy()
    output_df[SSQ_RAW_NAUSEA_COL] = raw_n
    output_df[SSQ_RAW_OCULOMOTOR_COL] = raw_o
    output_df[SSQ_RAW_DISORIENTATION_COL] = raw_d
    output_df[SSQ_NAUSEA_SCORE_COL] = nausea_score
    output_df[SSQ_OCULOMOTOR_SCORE_COL] = oculomotor_score
    output_df[SSQ_DISORIENTATION_SCORE_COL] = disorientation_score
    output_df[SSQ_TOTAL_SCORE_COL] = total_score
    output_df[SSQ_5LEVEL_CLASS_COL] = fivelevel_class
    output_df[SSQ_3CLASS_LABEL_COL] = threeclass_label
    output_df[SSQ_3CLASS_TARGET_COL] = threeclass_target

    ordered_columns = FEATURE_COLUMNS + ssq_item_columns(build) + [
        SSQ_RAW_NAUSEA_COL,
        SSQ_RAW_OCULOMOTOR_COL,
        SSQ_RAW_DISORIENTATION_COL,
        SSQ_NAUSEA_SCORE_COL,
        SSQ_OCULOMOTOR_SCORE_COL,
        SSQ_DISORIENTATION_SCORE_COL,
        SSQ_TOTAL_SCORE_COL,
        SSQ_5LEVEL_CLASS_COL,
        SSQ_3CLASS_LABEL_COL,
        SSQ_3CLASS_TARGET_COL,
    ]
    output_df = output_df.loc[:, ordered_columns]

    output_metadata_df = metadata_df.copy()
    for col, values in [
        (SSQ_RAW_NAUSEA_COL, raw_n),
        (SSQ_RAW_OCULOMOTOR_COL, raw_o),
        (SSQ_RAW_DISORIENTATION_COL, raw_d),
        (SSQ_NAUSEA_SCORE_COL, nausea_score),
        (SSQ_OCULOMOTOR_SCORE_COL, oculomotor_score),
        (SSQ_DISORIENTATION_SCORE_COL, disorientation_score),
        (SSQ_TOTAL_SCORE_COL, total_score),
        (SSQ_5LEVEL_CLASS_COL, fivelevel_class),
        (SSQ_3CLASS_LABEL_COL, threeclass_label),
        (SSQ_3CLASS_TARGET_COL, threeclass_target),
    ]:
        output_metadata_df[col] = values.to_numpy()

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(dataset_path, index=False)
    output_metadata_df.to_csv(metadata_path, index=False)

    return dataset_path, metadata_path, output_df, output_metadata_df


def build_distribution_exports(*, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_scores: list[pd.DataFrame] = []
    all_fivelevel: list[pd.DataFrame] = []
    all_threeclass: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for build in ("A", "B"):
        _, _, df, meta = build_ssq_3class_dataset(build, headfeatures_dir=headfeatures_dir, overwrite=False)

        score_df = meta[["row_index", "score_pid", "matched_tracking_pid", "tracking_file"]].copy()
        score_df.insert(0, "build", build)
        for col in [
            SSQ_RAW_NAUSEA_COL,
            SSQ_RAW_OCULOMOTOR_COL,
            SSQ_RAW_DISORIENTATION_COL,
            SSQ_NAUSEA_SCORE_COL,
            SSQ_OCULOMOTOR_SCORE_COL,
            SSQ_DISORIENTATION_SCORE_COL,
            SSQ_TOTAL_SCORE_COL,
            SSQ_5LEVEL_CLASS_COL,
            SSQ_3CLASS_LABEL_COL,
            SSQ_3CLASS_TARGET_COL,
        ]:
            score_df[col] = df[col].to_numpy()
        all_scores.append(score_df)

        fivelevel_dist, fivelevel_summary = summarize_distribution(
            score_df[SSQ_5LEVEL_CLASS_COL],
            build,
            FIVELEVEL_CLASSES,
            SSQ_5LEVEL_CLASS_COL,
        )
        threeclass_dist, threeclass_summary = summarize_distribution(
            score_df[SSQ_3CLASS_LABEL_COL],
            build,
            THREECLASS_CLASSES,
            SSQ_3CLASS_LABEL_COL,
        )
        all_fivelevel.append(fivelevel_dist)
        all_threeclass.append(threeclass_dist)

        summary_rows.append(
            {
                "build": build,
                "n_samples": int(len(score_df)),
                "mean_ssq_total": float(score_df[SSQ_TOTAL_SCORE_COL].mean()),
                "median_ssq_total": float(score_df[SSQ_TOTAL_SCORE_COL].median()),
                "std_ssq_total": float(score_df[SSQ_TOTAL_SCORE_COL].std()),
                "min_ssq_total": float(score_df[SSQ_TOTAL_SCORE_COL].min()),
                "q1_ssq_total": float(score_df[SSQ_TOTAL_SCORE_COL].quantile(0.25)),
                "q3_ssq_total": float(score_df[SSQ_TOTAL_SCORE_COL].quantile(0.75)),
                "max_ssq_total": float(score_df[SSQ_TOTAL_SCORE_COL].max()),
                "mean_ssq_nausea": float(score_df[SSQ_NAUSEA_SCORE_COL].mean()),
                "mean_ssq_oculomotor": float(score_df[SSQ_OCULOMOTOR_SCORE_COL].mean()),
                "mean_ssq_disorientation": float(score_df[SSQ_DISORIENTATION_SCORE_COL].mean()),
                **fivelevel_summary,
                **threeclass_summary,
            }
        )

    scores_df = pd.concat(all_scores, ignore_index=True)
    fivelevel_df = pd.concat(all_fivelevel, ignore_index=True)
    threeclass_df = pd.concat(all_threeclass, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    scores_df.to_csv(SCORES_OUTPUT_PATH, index=False)
    fivelevel_df.to_csv(FIVELEVEL_DIST_OUTPUT_PATH, index=False)
    threeclass_df.to_csv(THREECLASS_DIST_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    return scores_df, fivelevel_df, threeclass_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera datasets 3 classes de SSQ e exporta distribuicoes por build."
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
        help="Sobrescreve os arquivos 3 classes caso ja existam.",
    )
    parser.add_argument(
        "--headfeatures-dir",
        default=str(HEADFEATURES_DIR),
        help="Diretorio com os datasets originais de SSQ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builds = ("A", "B") if str(args.build).lower() == "all" else (normalize_build(args.build),)

    for build in builds:
        dataset_path, metadata_path, df, _ = build_ssq_3class_dataset(
            build,
            headfeatures_dir=args.headfeatures_dir,
            overwrite=args.overwrite,
        )
        counts = df[SSQ_3CLASS_LABEL_COL].value_counts().reindex(THREECLASS_CLASSES, fill_value=0)
        print(
            f"Build {build}: {dataset_path.name} | rows={len(df)} | "
            f"negligible={int(counts['negligible'])} | "
            f"intermediate={int(counts['intermediate'])} | "
            f"very_problematic={int(counts['very_problematic'])}"
        )
        print(f"Metadata: {metadata_path.name}")

    _, _, _, summary_df = build_distribution_exports(headfeatures_dir=args.headfeatures_dir)
    print()
    print(f"Saved: {SCORES_OUTPUT_PATH}")
    print(f"Saved: {FIVELEVEL_DIST_OUTPUT_PATH}")
    print(f"Saved: {THREECLASS_DIST_OUTPUT_PATH}")
    print(f"Saved: {SUMMARY_OUTPUT_PATH}")
    print()
    print(summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
