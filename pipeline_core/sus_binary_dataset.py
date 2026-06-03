from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_core.fast_tracking_ssq_dataset import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
HEADFEATURES_DIR = ROOT / "headfeatures_data"
SUS_SCORE_COL = "sus_score"
SUS_BINARY_TARGET_COL = "sus_not_acceptable_target"
SUS_ACCEPTABLE_THRESHOLD = 70.0


def normalize_build(build: str) -> str:
    build = build.upper().strip()
    if build not in {"A", "B"}:
        raise ValueError(f"Build invalido: {build}. Use 'A' ou 'B'.")
    return build


def source_dataset_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSSUS_Build{build}.xlsx"


def source_metadata_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSSUS_Build{build}_metadata.csv"


def binary_dataset_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSSUSBinary_Build{build}.xlsx"


def binary_metadata_path(build: str, headfeatures_dir: str | Path = HEADFEATURES_DIR) -> Path:
    build = normalize_build(build)
    return Path(headfeatures_dir) / f"HeadFeaturesVSSUSBinary_Build{build}_metadata.csv"


def sus_item_columns(build: str) -> list[str]:
    build = normalize_build(build)
    return [f"{build}-SUS{i}" for i in range(1, 11)]


def compute_sus_score(df: pd.DataFrame, build: str) -> pd.Series:
    build = normalize_build(build)
    positive = [f"{build}-SUS{i}" for i in (1, 3, 5, 7, 9)]
    negative = [f"{build}-SUS{i}" for i in (2, 4, 6, 8, 10)]
    return (df[positive].sub(1).sum(axis=1) + (5 - df[negative]).sum(axis=1)) * 2.5


def build_binary_target(sus_score: pd.Series) -> pd.Series:
    return sus_score.le(SUS_ACCEPTABLE_THRESHOLD).astype(int)


def build_sus_binary_dataset(
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

    sus_score = compute_sus_score(source_df, build).round(4)
    binary_target = build_binary_target(sus_score)

    output_df = source_df.copy()
    output_df[SUS_SCORE_COL] = sus_score
    output_df[SUS_BINARY_TARGET_COL] = binary_target

    ordered_columns = FEATURE_COLUMNS + sus_item_columns(build) + [SUS_SCORE_COL, SUS_BINARY_TARGET_COL]
    output_df = output_df.loc[:, ordered_columns]

    output_metadata_df = metadata_df.copy()
    output_metadata_df[SUS_SCORE_COL] = sus_score.to_numpy()
    output_metadata_df[SUS_BINARY_TARGET_COL] = binary_target.to_numpy()

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(dataset_path, index=False)
    output_metadata_df.to_csv(metadata_path, index=False)

    return dataset_path, metadata_path, output_df, output_metadata_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera datasets binarios de SUS por build, com score SUS e target 1=nao aceitavel."
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
        help="Diretorio com os datasets originais de SUS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builds = ("A", "B") if str(args.build).lower() == "all" else (normalize_build(args.build),)

    for build in builds:
        dataset_path, metadata_path, df, metadata_df = build_sus_binary_dataset(
            build,
            headfeatures_dir=args.headfeatures_dir,
            overwrite=args.overwrite,
        )
        positive_count = int(df[SUS_BINARY_TARGET_COL].sum())
        print(
            f"Build {build}: {dataset_path.name} | "
            f"rows={len(df)} | positives={positive_count} | "
            f"positive_pct={positive_count / len(df):.2%}"
        )
        print(f"Metadata: {metadata_path.name} | rows={len(metadata_df)}")


if __name__ == "__main__":
    main()
