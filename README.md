# Head Metrics Classification Notebooks

This folder is a **paper-reference companion workspace** for the study:

**From Head Motion to Subjective VR Outcomes: A Reproducible Metric Framework and Benchmark on the FAST Dataset**

The intention is the same as in repositories that accompany research papers: provide a reusable, inspectable, and reproducible implementation of the main tabular head-metrics pipeline used in the study, while also making it easier for other users to adapt the workflow to their own datasets.

This folder is a portable workspace for users who want to:

1. convert their own head-metrics dataset into the same 47-feature schema used in the FAST experiments;
2. run configurable classification pipelines with multiple models, preprocessing options, imbalance strategies, and threshold-search strategies.

The code is intentionally separated from the FAST-specific scripts in the repository. It expects tabular head metrics, not raw time-series tracking files.

## Relation To The Paper

This `notebook/` folder should be understood as a **reference implementation layer** for the paper, not as a full reproduction of every script in the repository.

Its role is to expose, in a more portable format:

- the **47-feature head-metrics schema** used across the experiments;
- the **classification workflow** built around repeated stratified cross-validation;
- the handling of **imbalance strategies** such as class weights, random undersampling, `SMOTE`, and `NearMiss`;
- the handling of **threshold-search strategies** such as default threshold, `roc_gmean`, and precision-recall based search;
- a simple way to apply the same logic to **new external datasets**.

In that sense, this folder follows the style commonly used by paper repositories:

- the main repository keeps the research code and experiment history;
- this subfolder exposes a cleaner entry point for reuse;
- notebooks provide an interactive path for inspection and adaptation;
- scripts provide the same logic in reusable and automatable form.

## Directory Layout

- `data/original/`
  Input spreadsheets used by the notebooks. The repository ships with a bundled FAST SUS demo pair.
- `data/generated/`
  Standardized spreadsheets produced by the preparation notebook or script.
- `outputs/`
  Model-selection outputs, metrics, predictions, and serialized artifacts.
- `configs/`
  JSON files for dataset mapping and classification-pipeline configuration.
- `scripts/`
  Reusable Python code called by the notebooks and CLI commands.

All notebook defaults use paths relative to `notebook/`. This is intentional so the folder remains portable when pushed to GitHub or cloned elsewhere.

## Files

- `01_prepare_head_metrics_dataset.ipynb`
  Standardizes input spreadsheets from `data/original/` into the expected schema and writes them to `data/generated/`.
- `02_run_classification_pipeline.ipynb`
  Runs model selection on a training set from `data/generated/` and evaluates the selected pipeline on a test set.
- `scripts/head_metrics_schema.py`
  Defines the 47 input variables and feature families.
- `scripts/transform_head_metrics.py`
  CLI and Python functions for column mapping, validation, and standardized output.
- `scripts/classification_pipeline.py`
  Configurable classification runner with CV, imbalance handling, threshold tuning, and test evaluation.
- `configs/transform_config_template.json`
  Template for mapping user column names to the expected feature names.
- `configs/classification_config_template.json`
  Template for model/preprocessing/imbalance/threshold experiment configuration.
- `configs/transform_config_fast_sus_demo.json`
  Demo mapping config used by the default preparation notebook.
- `configs/classification_config_fast_sus_demo.json`
  Demo experiment config used by the default classification notebook.

## Bundled Demo Data

The repository includes a small ready-to-run example:

- `data/original/FAST_SUS_BuildA_original.xlsx`
- `data/original/FAST_SUS_BuildB_original.xlsx`

These are not raw tracking files. They are already tabular head-metrics spreadsheets, meant to demonstrate the transformation and classification workflow with the same style of inputs used throughout the paper pipeline.

In the bundled demo:

- the feature names already match the FAST schema;
- the transform step is effectively an identity mapping;
- the target column is `sus_not_acceptable_target`;
- `sus_score` is preserved as a passthrough metadata column in the generated files.

For a new external dataset, replace the files in `data/original/` or point the notebooks to another location and edit the JSON mapping accordingly.

## Scope

This folder is designed for **classification experiments on already-computed head metrics**.

It does **not**:

- extract features from raw tracking streams;
- recreate every experiment reported in the repository;
- replace the full experiment scripts used for the paper.

Instead, it provides a compact and configurable benchmark-style pipeline that is aligned with the paper methodology and can be reused on user-provided datasets.

## Expected Dataset Format

The standardized dataset must contain:

- the 47 feature columns listed in `scripts/head_metrics_schema.py`;
- a classification target column, for example `target`;
- optionally metadata columns such as `participant_id`, `build`, or score columns preserved as passthrough fields.

For binary classification, the target must be encoded as:

- `0`: negative/majority/reference class;
- `1`: positive class.

For multiclass classification, the target must be integer encoded, for example `0`, `1`, `2`.

## Command-Line Usage

Standardize a train dataset:

```powershell
python notebook\scripts\transform_head_metrics.py --input notebook\data\original\FAST_SUS_BuildA_original.xlsx --output notebook\data\generated\FAST_SUS_BuildA_standardized.xlsx --config notebook\configs\transform_config_fast_sus_demo.json
```

Standardize a test dataset:

```powershell
python notebook\scripts\transform_head_metrics.py --input notebook\data\original\FAST_SUS_BuildB_original.xlsx --output notebook\data\generated\FAST_SUS_BuildB_standardized.xlsx --config notebook\configs\transform_config_fast_sus_demo.json
```

Run the classification pipeline:

```powershell
python notebook\scripts\classification_pipeline.py --config notebook\configs\classification_config_fast_sus_demo.json
```

To adapt the workflow to your own data, duplicate the demo configs or start from the two template files in `configs/`.

## Methodological Rules

- Cross-validation and model selection happen only on the training dataset.
- Resampling strategies are applied only inside training folds.
- Thresholds are selected from training-fold predictions and then transferred to the test set.
- The test dataset is evaluated only once after pipeline selection.

This mirrors the Build A -> Build B logic used in the main FAST experiments.

## Reference

This folder is associated with the paper:

**From Head Motion to Subjective VR Outcomes: A Reproducible Metric Framework and Benchmark on the FAST Dataset**

If you use this workflow in derivative experiments or adaptations, it should be cited as the implementation reference associated with that study.
