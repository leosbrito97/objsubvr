# objsubvr

Reference pipeline for the paper:

**From Head Yaw to HMD Pose: A Reproducible Benchmark of Subjective VR Outcome Recoverability**

This repository is organized around one reproducible workflow:

```text
FAST-style raw data
  -> HMD pose feature extraction
  -> selected article classification pipeline
  -> reproducible output tables
```

The repository is not intended to preserve the full internal experiment history. It exposes the final paper pipeline in a form that external readers can run, inspect, and adapt.

## Repository Structure

```text
configs/
  feature_extraction_fast.json      # full FAST-style feature extraction
  feature_extraction_demo.json      # included demo feature extraction
  pipeline_article_best.json        # selected best model config per questionnaire
  examples/                         # auxiliary configs for demos/templates

scripts/
  extract_features.py               # raw tracking + scores -> feature tables
  run_pipeline.py                   # feature tables -> article benchmark outputs
  feature_engineering_pipeline.py   # reusable feature-extraction helpers
  classification_pipeline.py        # generic classification utility
  transform_head_metrics.py         # schema-alignment utility

data/raw/fast_demo/                 # small included FAST-style demo subset
docs/feature_extraction_parameters.md
outputs/                            # generated outputs
```

The public reproduction path uses only:

```text
scripts/extract_features.py
scripts/run_pipeline.py
configs/feature_extraction_fast.json
configs/pipeline_article_best.json
```

## Install

```powershell
pip install -r requirements.txt
```

Optional model packages such as LightGBM, XGBoost, CatBoost, and imbalanced-learn are listed in `requirements.txt` because the selected article configuration uses SVM, LightGBM, SMOTE, NearMiss, and undersampling.

## Inputs

For full article reproduction, provide FAST-style data with this layout:

```text
FAST-Dataset/
  FAST Scores.xlsx
  Tracking/
    <participant>_BuildA_<timestamp>.csv
    <participant>_BuildB_<timestamp>.csv
```

The expected tracking columns are:

```text
Timestamp
Head_position_x
Head_position_y
Head_position_z
Head_quat_x
Head_quat_y
Head_quat_z
Head_quat_w
```

The full article numbers require the full FAST participant set. The included `data/raw/fast_demo/` subset is only for checking that feature extraction runs.

## Smoke Test

Run feature extraction on the included demo subset:

```powershell
python scripts\extract_features.py --config configs\feature_extraction_demo.json
```

This writes demo feature tables to:

```text
outputs/demo/headfeatures_data/
```

The demo subset should not be used to compare against article metrics.

## Full Article Pipeline

### 1. Extract Features

Place the full FAST-style data under `FAST-Dataset/`, or edit paths in:

```text
configs/feature_extraction_fast.json
```

Then run:

```powershell
python scripts\extract_features.py --config configs\feature_extraction_fast.json
```

This generates:

```text
headfeatures_data/
  HeadFeaturesVSSUSBinary_BuildA.xlsx
  HeadFeaturesVSSUSBinary_BuildB.xlsx
  HeadFeaturesVSTLXBinary_BuildA.xlsx
  HeadFeaturesVSTLXBinary_BuildB.xlsx
  HeadFeaturesVSSPESBinary_BuildA.xlsx
  HeadFeaturesVSSPESBinary_BuildB.xlsx
  HeadFeaturesVSSSQ3Class_BuildA.xlsx
  HeadFeaturesVSSSQ3Class_BuildB.xlsx
  *_metadata.csv
  feature_extraction_manifest.csv
```

### 2. Run The Selected Article Pipeline

The final selected configuration is stored in:

```text
configs/pipeline_article_best.json
```

It defines:

- A+B pooled participant-grouped 7-fold evaluation;
- Pearson redundancy feature reduction;
- the selected model per questionnaire;
- the selected preprocessing strategy;
- the selected imbalance strategy;
- the selected threshold strategy for binary targets;
- the selected hyperparameters.

Run:

```powershell
python scripts\run_pipeline.py --config configs\pipeline_article_best.json
```

Outputs are written to:

```text
outputs/article_best/
```

Important output files:

```text
outputs/article_best/article_best_results.md
outputs/article_best/article_best_summary.csv
outputs/article_best/article_best_selected_by_questionnaire.csv
outputs/article_best/<task>/<task>_selected_results.csv
outputs/article_best/<task>/<task>_test_predictions.csv
outputs/article_best/<task>/<task>_feature_reduction_details.csv
outputs/article_best/<task>/<task>_participant_fold_assignment.csv
```

## Selected Configurations

The selected article configurations are encoded in `configs/pipeline_article_best.json`:

| Task | Model | Preprocessing | Imbalance | Threshold |
|:--|:--|:--|:--|:--|
| SUS binary | SVM linear, `C=1.0` | standard scaling | NearMiss | ROC G-mean |
| NASA-TLX binary | SVM linear, `C=0.25` | power transform + standard scaling | SMOTE | PR ideal distance |
| SPES binary | LightGBM, 150 estimators | median imputation | class weights | ROC G-mean |
| SSQ 3-class | LightGBM, 150 estimators | winsorization + imputation | undersampling | not applicable |

This script does not rerun the full model-search experiment. It evaluates the final configurations selected during the manuscript experiments.

## Methodological Safeguards

The article pipeline is designed to reduce information leakage:

- Build A and Build B are pooled for each questionnaire task.
- Splits are grouped by participant identifier.
- Build A and Build B observations from the same participant stay in the same split partition.
- Each split uses training, calibration, and test partitions.
- Preprocessing is fitted only on training data.
- Pearson feature reduction is fitted only on training data.
- Imbalance handling is applied only during model fitting.
- Binary thresholds are selected only on the calibration fold.
- The test fold is used only for final evaluation.

## Feature-Extraction Parameters

Exact feature-extraction parameters are documented in:

```text
docs/feature_extraction_parameters.md
```

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

## Adapting The Pipeline

External users can adapt the pipeline in two ways:

1. Use FAST-style raw files and edit `configs/feature_extraction_fast.json`.
2. Use precomputed tabular head metrics and align them to the expected schema with `scripts/transform_head_metrics.py`.

The required feature schema is defined in:

```text
scripts/head_metrics_schema.py
```

Auxiliary templates and older demo configs are kept under:

```text
configs/examples/
```

They are not part of the official paper reproduction path.

## Notebooks

The notebooks are lightweight demos:

```text
01_prepare_head_metrics_dataset.ipynb
02_run_classification_pipeline.ipynb
```

They are useful for exploring the workflow interactively. The official reproduction path is the command-line pipeline described above.

## Citation

If this repository is useful, cite:

**From Head Yaw to HMD Pose: A Reproducible Benchmark of Subjective VR Outcome Recoverability**
