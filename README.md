# *From Head Yaw to HMD Pose

Reference pipeline for the paper **From Head Yaw to HMD Pose: A Reproducible Benchmark of Subjective VR Outcome Recoverability**.

This repository exposes the final reproducible pipeline used by the paper. The workflow is intentionally simple:

```text
FAST-style raw data
  -> feature extraction
  -> selected article pipeline
  -> output tables
```

The internal experiment history is not part of this repository. The goal here is to make the final pipeline easy for external readers, reviewers, and researchers to run.

## Index

| Section | Description |
|:--|:--|
| [Repository Layout](#repository-layout) | Where code, configs, notebooks, data, and outputs live. |
| [Install](#install) | Python dependencies. |
| [Input Data](#input-data) | Required FAST-style data layout and tracking columns. |
| [Feature Extraction](#feature-extraction) | Raw scores/tracking files to questionnaire feature tables. |
| [Article Pipeline](#article-pipeline) | Selected article configuration and evaluation command. |
| [Configuration Files](#configuration-files) | Main JSON configs and auxiliary examples. |
| [Outputs](#outputs) | Files produced by the selected pipeline. |
| [Selected Model Configurations](#selected-model-configurations) | Final model choices per questionnaire. |
| [Methodological Safeguards](#methodological-safeguards) | Leakage-control decisions used by the pipeline. |
| [Feature-Extraction Parameters](#feature-extraction-parameters) | Parameter values used to compute tracking features. |
| [Notebooks](#notebooks) | Interactive explanatory material. |
| [Adapting The Pipeline](#adapting-the-pipeline) | How to reuse the pipeline with other data. |
| [Citation](#citation) | Paper citation placeholder. |

## Repository Layout

```text
configs/
  feature_extraction_fast.json
  feature_extraction_demo.json
  pipeline_article_best.json
  examples/

pipeline_core/
  fast_tracking_ssq_dataset.py
  feature_families.py
  sus_binary_dataset.py
  tlx_binary_dataset.py
  spes_binary_dataset.py
  ssq_3class_dataset.py

scripts/
  extract_features.py
  run_pipeline.py
  feature_engineering_pipeline.py
  classification_pipeline.py
  transform_head_metrics.py
  head_metrics_schema.py

notebooks/
  01_prepare_head_metrics_dataset.ipynb
  02_run_classification_pipeline.ipynb

data/raw/fast_demo/
docs/
outputs/
```

The intended public commands are:

| Step | Command |
|:--|:--|
| Extract features | `python scripts\extract_features.py --config configs\feature_extraction_fast.json` |
| Run article pipeline | `python scripts\run_pipeline.py --config configs\pipeline_article_best.json` |

Files under `pipeline_core/` are internal Python modules imported by the commands above. They are not meant to be executed directly.

## Install

```powershell
pip install -r requirements.txt
```

The selected article pipeline uses scikit-learn, imbalanced-learn, and LightGBM. Other optional model libraries are kept in the requirements because the generic classification utility can use them.

## Input Data

For full article reproduction, provide FAST-style data with this layout:

```text
FAST-Dataset/
  FAST Scores.xlsx
  Tracking/
    <participant>_BuildA_<timestamp>.csv
    <participant>_BuildB_<timestamp>.csv
```

Expected tracking columns:

| Column |
|:--|
| `Timestamp` |
| `Head_position_x` |
| `Head_position_y` |
| `Head_position_z` |
| `Head_quat_x` |
| `Head_quat_y` |
| `Head_quat_z` |
| `Head_quat_w` |

The full article metrics require the full FAST participant set. The included `data/raw/fast_demo/` subset is only a small example dataset for checking file format and feature extraction.

## Feature Extraction

Edit paths in:

```text
configs/feature_extraction_fast.json
```

Then run:

```powershell
python scripts\extract_features.py --config configs\feature_extraction_fast.json
```

This creates questionnaire-specific feature tables in `headfeatures_data/`.

Generated tables include:

| Output |
|:--|
| `HeadFeaturesVSSUSBinary_BuildA.xlsx` |
| `HeadFeaturesVSSUSBinary_BuildB.xlsx` |
| `HeadFeaturesVSTLXBinary_BuildA.xlsx` |
| `HeadFeaturesVSTLXBinary_BuildB.xlsx` |
| `HeadFeaturesVSSPESBinary_BuildA.xlsx` |
| `HeadFeaturesVSSPESBinary_BuildB.xlsx` |
| `HeadFeaturesVSSSQ3Class_BuildA.xlsx` |
| `HeadFeaturesVSSSQ3Class_BuildB.xlsx` |
| `*_metadata.csv` |
| `feature_extraction_manifest.csv` |

## Article Pipeline

The final selected pipeline is defined in:

```text
configs/pipeline_article_best.json
```

Run:

```powershell
python scripts\run_pipeline.py --config configs\pipeline_article_best.json
```

This evaluates the selected article configuration for each questionnaire. It does not rerun the full model-search history.

## Configuration Files

Main configs:

| File | Purpose |
|:--|:--|
| `configs/feature_extraction_fast.json` | Full FAST-style feature extraction. |
| `configs/feature_extraction_demo.json` | Example feature extraction using the included demo subset. |
| `configs/pipeline_article_best.json` | Final selected article pipeline configuration. |

Auxiliary templates and older demo configs are stored in `configs/examples/`. They are not part of the official article reproduction path.

## Outputs

The selected article pipeline writes to `outputs/article_best/`.

Important files:

| Output |
|:--|
| `outputs/article_best/article_best_results.md` |
| `outputs/article_best/article_best_summary.csv` |
| `outputs/article_best/article_best_selected_by_questionnaire.csv` |
| `outputs/article_best/<task>/<task>_selected_results.csv` |
| `outputs/article_best/<task>/<task>_test_predictions.csv` |
| `outputs/article_best/<task>/<task>_feature_reduction_details.csv` |
| `outputs/article_best/<task>/<task>_participant_fold_assignment.csv` |

## Selected Model Configurations

The selected article configurations are encoded in `configs/pipeline_article_best.json`.

| Task | Model | Preprocessing | Imbalance | Threshold |
|:--|:--|:--|:--|:--|
| SUS binary | SVM linear, `C=1.0` | standard scaling | NearMiss | ROC G-mean |
| NASA-TLX binary | SVM linear, `C=0.25` | power transform + standard scaling | SMOTE | PR ideal distance |
| SPES binary | LightGBM, 150 estimators | median imputation | class weights | ROC G-mean |
| SSQ 3-class | LightGBM, 150 estimators | winsorization + imputation | undersampling | not applicable |

## Methodological Safeguards

The pipeline follows the article evaluation protocol:

| Safeguard | Description |
|:--|:--|
| A+B pooling | Build A and Build B are pooled for each questionnaire. |
| Participant grouping | Splits are grouped by participant identifier. |
| Paired build assignment | Build A and Build B observations from the same participant stay in the same split partition. |
| Split roles | Each split uses training, calibration, and test partitions. |
| Preprocessing isolation | Preprocessing is fitted only on training data. |
| Feature-reduction isolation | Pearson feature reduction is fitted only on training data. |
| Imbalance handling | Imbalance handling is applied only during model fitting. |
| Threshold selection | Binary thresholds are selected only on the calibration fold. |
| Final testing | The test fold is used only for final evaluation. |

## Feature-Extraction Parameters

Exact feature-extraction parameters are documented in `docs/feature_extraction_parameters.md`.

Key values:

| Parameter | Value |
|:--|:--|
| Stationary-speed threshold | `0.02 m/s` |
| Yaw turn threshold | `0.05 deg` |
| Downward-pitch threshold | `-10.0 deg` |
| Extreme-pitch threshold | `30.0 deg` |
| Yaw-pitch entropy grid | `(24, 20)` bins |
| Quaternion-to-Euler convention | `Rotation.from_quat([x, y, z, w])`, then `as_euler("yxz", degrees=True)` |
| Angle unwrapping | `np.unwrap` on yaw and pitch radians |
| Filtering/smoothing | no smoothing or low-pass filter |

## Notebooks

The notebooks are in `notebooks/`. They are explanatory material. The official reproduction path is the command-line pipeline in `scripts/`.

## Adapting The Pipeline

External users can adapt the pipeline in two ways:

1. Use FAST-style raw files and edit `configs/feature_extraction_fast.json`.
2. Use precomputed tabular head metrics and align them to the expected schema with `scripts/transform_head_metrics.py`.

The expected feature schema is defined in `scripts/head_metrics_schema.py`.

## Citation

If this repository is useful, cite:

**From Head Yaw to HMD Pose: A Reproducible Benchmark of Subjective VR Outcome Recoverability**
