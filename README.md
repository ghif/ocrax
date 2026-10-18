# OCRax

OCRax is a research codebase for optical character recognition (OCR), focused on PP-OCRv6 text recognition and the RxHandBD handwritten-word dataset. It provides JAX/Flax inference and fine-tuning, PaddleOCR evaluation, checkpoint conversion, cross-framework parity checks, and a Colab training workflow.

The project is organized as a Python package under `src/ocrax`. Its command-line tools share the `ocrax-` prefix. This guide introduces the project, explains how to run the supported workflows, and describes how to collaborate on changes and experiments.

## Quick links

[Quick start](#quick-start) · [Project structure](#project-structure) · [Prepare RxHandBD manifests](#prepare-rxhandbd-manifests) · [Inference and evaluation](#inference-and-evaluation) · [Fine-tune the JAX recognizer](#fine-tune-the-jax-recognizer) · [Continue from a checkpoint](#continue-from-a-checkpoint) · [Convert checkpoints and check parity](#convert-checkpoints-and-check-parity) · [Train on Colab](#train-on-colab) · [Commands](#commands) · [Collaborating on OCRax](#collaborating-on-ocrax) · [Research notes and limitations](#research-notes-and-limitations)

## Quick start

### Requirements

- Conda with Python 3.13.x.
- A JAX-compatible CPU or accelerator runtime. JAX installation steps vary by platform; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
- Optional: PaddlePaddle and PaddleOCR for Paddle inference, model export, and parity checks.
- Optional: Google Cloud authentication and the Colab CLI for remote training.

### Install

From the repository root, create and activate the Conda environment, then install OCRax in editable mode:

```bash
conda create -n ocrax python=3.13
conda activate ocrax
python --version  # should report Python 3.13.x
python -m pip install --upgrade pip
python -m pip install -e .
```

Install optional dependencies for the workflows you plan to use:

```bash
# Paddle inference, export, and parity
python -m pip install -e ".[paddle]"

# Google authentication and Cloud Storage support
python -m pip install -e ".[cloud]"

# Both groups
python -m pip install -e ".[paddle,cloud]"
```

For a quick check that installation registered the commands:

```bash
ocrax-train --help
ocrax-evaluate-jax --help
```

Commands can also be run as Python modules, for example: `python -m ocrax.cli.train --help`. On a machine where JAX selects an unavailable accelerator backend, set `JAX_PLATFORMS=cpu` before running a JAX command.

## Project structure

```text
ocrax/
├── artifacts/checkpoints/      # Local, Git-ignored Orbax checkpoints
├── data/manifests/             # Local, Git-ignored train/validation manifests
├── docs/                       # Evaluation reports and research notes
├── scripts/                    # Colab runtime bootstrap
├── src/ocrax/
│   ├── application/            # Training, evaluation, conversion, parity flows
│   ├── cli/                    # User-facing command-line interfaces
│   ├── domain/                 # Shared metrics and domain rules
│   └── infrastructure/         # Data, models, checkpoints, cloud, reporting
├── tests/                      # Unit and mocked workflow tests
└── pyproject.toml              # Package metadata and command registration
```

## Prepare RxHandBD manifests

The dataset is expected to have this layout:

```text
RxHandBD/
├── Train_Label.csv
├── Train_Set/
├── Test_Labels.csv
└── Test_Set/
```

Generate train and validation manifests from the training labels:

```bash
ocrax-prepare-data \
  --dataset-root /path/to/RxHandBD \
  --output-dir data/manifests \
  --validation-fraction 0.15 \
  --seed 20261017
```

The command writes `train.jsonl`, `validation.jsonl`, and `split_metadata.json`. It reads `Train_Label.csv` only; it does not read the test labels. The split is sample-level because the supplied labels do not identify writers or source documents. As a result, related samples may appear in both training and validation data.

Prepared manifests are dataset-derived and are not included in the Git repository. Generate them locally with the command above before training. Manifest JSONL files contain image paths and transcript text; the generated `data/manifests/` directory is ignored by Git. Run preparation from the repository root when using the default relative output path.

## Inference and evaluation

The evaluation commands run predictions over the RxHandBD test set and write one row per image to a CSV. Evaluation intentionally reads `Test_Labels.csv` to calculate metrics. Fine-tuning and manifest preparation do not use test labels.

### Paddle PP-OCRv6

`ocrax-evaluate` uses PaddleOCR's `TextRecognition` API and the `PP-OCRv6_medium_rec` model. PaddleOCR may download the model on first use. Install the Paddle extras, then run:

```bash
ocrax-evaluate \
  --dataset-root /path/to/RxHandBD \
  --device cpu \
  --batch-size 32 \
  --output outputs/rxhandbd_ppocrv6_predictions.csv
```

Use `--device gpu:0` when Paddle is configured for GPU inference.

### JAX PP-OCRv6

`ocrax-evaluate-jax` runs the local Flax NNX model using an Orbax checkpoint. It also needs the local PP-OCRv6 model directory to load the character dictionary from `inference.yml`.

```bash
JAX_PLATFORMS=cpu ocrax-evaluate-jax \
  --dataset-root /path/to/RxHandBD \
  --model-dir /path/to/PP-OCRv6_medium_rec \
  --checkpoint artifacts/checkpoints/ppocrv6_medium_rec_orbax \
  --batch-size 32 \
  --output outputs/rxhandbd_ppocrv6_jax_predictions.csv
```

The CLI defaults to the checkpoint path shown above, but checkpoint files are not included in the Git repository. Generate or obtain the checkpoint locally before running evaluation, and supply `--checkpoint` if it is stored elsewhere. Supply `--model-dir` when the model is not available at the conventional local model path.

### PaddleOCR-VL

`ocrax-evaluate-vl` runs PaddleOCR-VL-0.9B on word crops. It requires a local model checkpoint and does not download a missing checkpoint. Install its optional dependencies in the active environment:

```bash
python -m pip install 'paddleocr>=3.7' 'paddlex[ocr]==3.7.2'
ocrax-evaluate-vl \
  --dataset-root /path/to/RxHandBD \
  --model-dir /path/to/PaddleOCR-VL-0.9B \
  --device cpu \
  --output outputs/rxhandbd_paddleocr_vl_predictions.csv
```

Use `--device gpu:0` when supported by your Paddle installation.

### Single-image Paddle inference from Python

For a quick reference prediction, use the same preprocessing and Paddle predictor adapter used by the parity workflow:

```python
from pathlib import Path

from PIL import Image

from ocrax.application.parity import load_paddle_predictor, paddle_forward
from ocrax.infrastructure.data.images import resize_rec_image

model_dir = Path("/path/to/PP-OCRv6_medium_rec")
image_path = Path("/path/to/word-crop.jpg")

with Image.open(image_path) as image:
    image_chw, _ = resize_rec_image(image)

predictor = load_paddle_predictor(model_dir)
probabilities = paddle_forward(predictor, image_chw[None, ...])
print(probabilities.shape)  # [1, 40, 18_710]
```

Install `.[paddle]` before running the example. The output contains CTC class probabilities; decode them with the character dictionary in `inference.yml`.

### Metrics and prediction CSV

Evaluators report raw and normalized exact-match accuracy, character error rate (CER), and word error rate (WER) or word-level accuracy (`1 - WER`). Their CSV files share this schema:

```text
image,ground_truth,prediction,confidence,correct
```

PaddleOCR-VL does not return comparable per-character confidence, so its confidence field is empty.

## Fine-tune the JAX recognizer

Training starts from an Orbax checkpoint and consumes the train and validation manifests. It never reads test labels. Choose a new output directory for each run; the trainer refuses to overwrite an existing directory.

```bash
JAX_PLATFORMS=cpu ocrax-train \
  --model-dir /path/to/PP-OCRv6_medium_rec \
  --checkpoint artifacts/checkpoints/ppocrv6_medium_rec_orbax \
  --train-manifest data/manifests/train.jsonl \
  --validation-manifest data/manifests/validation.jsonl \
  --output artifacts/checkpoints/my_rxhandbd_run \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --patience 3 \
  --min-delta 0.001 \
  --seed 20261017
```

The model directory must contain `inference.yml`. Each JSONL manifest row contains an image path and transcript. Defaults are defined by the CLI; use `ocrax-train --help` to review all options.

The current recipe uses the Flax NNX PP-OCRv6 medium CTC recognizer with Optax AdamW, global gradient norm clipping at `1.0`, weight decay `1e-5`, and brightness/contrast augmentation from `0.9` to `1.1`. BatchNorm running statistics are frozen to preserve the fused inference checkpoint semantics. The default learning rate is `1e-5`; early stopping selects the best validation CER using patience `3` and minimum significant improvement `0.001`.

This is a **CTC-only fine-tune**. The available inference checkpoint does not contain training-only NRTR auxiliary head weights, so this workflow does not reproduce the official CTC+NRTR training objective.

The output contains the best model-state checkpoint and `finetuning.json` with the run settings, manifest hashes, baseline and best validation metrics, and epoch history. Optimizer slots are not saved; continuation initializes a new optimizer from the saved model weights.

To evaluate the resulting model on the test set after model selection is complete, run:

```bash
JAX_PLATFORMS=cpu ocrax-evaluate-jax \
  --dataset-root /path/to/RxHandBD \
  --model-dir /path/to/PP-OCRv6_medium_rec \
  --checkpoint artifacts/checkpoints/my_rxhandbd_run \
  --output outputs/my_rxhandbd_predictions.csv
```

Use validation metrics for iterative decisions and reserve test metrics for final comparisons.

## Continue from a checkpoint

`ocrax-resume` wraps the same training recipe for local or Colab continuation. Continuation restores model weights only; it starts a new optimizer and epoch sequence rather than resuming optimizer state or an interrupted epoch.

Local continuation:

```bash
ocrax-resume local \
  --model-dir /path/to/PP-OCRv6_medium_rec \
  --checkpoint artifacts/checkpoints/my_rxhandbd_run \
  --output artifacts/checkpoints/my_rxhandbd_continuation \
  --epochs 3
```

Local resume uses batch size `8`, learning rate `1e-5`, and seed `20261017`. For full control over training options, invoke `ocrax-train` directly with the prior checkpoint and a new output directory.

## Convert checkpoints and check parity

The optional conversion workflow exports arrays from a local Paddle inference model, maps the parameters to the Flax NNX model, and writes an Orbax checkpoint. Install `.[paddle]` first.

```bash
ocrax-export-paddle \
  --model-dir /path/to/PP-OCRv6_medium_rec \
  --output /tmp/ppocrv6_params.npz

JAX_PLATFORMS=cpu ocrax-convert \
  --arrays /tmp/ppocrv6_params.npz \
  --output /tmp/ppocrv6_orbax
```

Compare Paddle and JAX inference on representative crops:

```bash
JAX_PLATFORMS=cpu ocrax-parity \
  --model-dir /path/to/PP-OCRv6_medium_rec \
  --checkpoint /tmp/ppocrv6_orbax \
  --atol 2e-4 \
  --rtol 2e-4 \
  --report outputs/parity-report.json \
  /path/to/crop-1.jpg /path/to/crop-2.jpg
```

Parity checks output shapes, logits, probabilities, and CTC timestep argmaxes. The default absolute and relative tolerances are both `2e-4`. A shape mismatch or failed numeric check causes the command to fail. The converted checkpoint used for the recorded parity run is not distributed in Git; generate it with the export and conversion steps above or provide a local checkpoint path.

## Train on Colab

The Colab launcher packages the OCRax source and manifests, stages inputs from Google Cloud Storage inside the runtime, runs the same trainer, uploads the checkpoint to a unique output prefix, and stops the session. Model and dataset contents are not downloaded to your local machine. Install `.[cloud]`, install the Colab CLI, and authenticate with Google Application Default Credentials (ADC):

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

The ADC account needs read access to the model/checkpoint and dataset objects, and permission to create objects under the output prefix. If ADC reports that no quota project is set, configure one your account can use:

```bash
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Launch a run:

```bash
ocrax-colab \
  --model-gcs gs://YOUR_BUCKET/model-dir \
  --dataset-gcs gs://YOUR_BUCKET/RxHandBD \
  --gpu T4 \
  --epochs 3
```

The model prefix must include `inference.yml` and an Orbax checkpoint named `ppocrv6_medium_rec_orbax`. By default, the run is written under `MODEL_GCS/finetuned-runs/<unique-run-id>`. Pass `--checkpoint-gcs` to use a different Orbax checkpoint prefix. GPU choices are `T4`, `L4`, `G4`, `A100`, and `H100`, subject to Colab availability.

Continue from a previous Colab checkpoint with:

```bash
ocrax-resume colab \
  --model-gcs gs://YOUR_BUCKET/model-dir \
  --checkpoint-gcs gs://YOUR_BUCKET/model-dir/finetuned-runs/RUN_ID \
  --dataset-gcs gs://YOUR_BUCKET/RxHandBD \
  --gpu T4 \
  --epochs 3
```

The resumed checkpoint is written to a new unique output prefix; earlier runs are preserved. Use `ocrax-colab --help` and `ocrax-resume --help` for options.

## Commands

| Command | Purpose | Dependency group |
| --- | --- | --- |
| `ocrax-prepare-data` | Prepare train and validation manifests | Base |
| `ocrax-train` | Fine-tune the JAX CTC recognizer | Base |
| `ocrax-resume` | Continue locally or on Colab | Base; Cloud for Colab |
| `ocrax-evaluate-jax` | Evaluate an Orbax checkpoint | Base |
| `ocrax-evaluate` | Evaluate Paddle PP-OCRv6 | Paddle |
| `ocrax-evaluate-vl` | Evaluate PaddleOCR-VL-0.9B | Paddle + PaddleX OCR extras |
| `ocrax-export-paddle` | Export local Paddle model arrays | Paddle |
| `ocrax-paddle-probe` | Inspect selected Paddle parameter mappings | Paddle |
| `ocrax-convert` | Convert exported arrays to an Orbax checkpoint | Base |
| `ocrax-parity` | Compare Paddle and JAX inference | Paddle |
| `ocrax-colab` | Launch remote Colab fine-tuning | Cloud + Colab CLI |

All commands accept `--help` to display their arguments and defaults.

## Collaborating on OCRax

OCRax is organized to make research workflows easier to review and reproduce. When contributing, keep changes close to the layer that owns the behavior:

- Put reusable recognition rules and metric definitions in `domain/`.
- Put end-to-end use cases in `application/`.
- Keep argument parsing and user-facing command behavior in `cli/`.
- Keep framework, filesystem, cloud, model, and serialization details in `infrastructure/`.
- Add or update focused tests alongside changes to behavior.
- Update this README or the relevant research note when commands, defaults, output formats, or experimental assumptions change.

### Reproducible experiments

For a training or evaluation change, include enough information for a collaborator to reproduce and interpret the result:

1. Record the command and effective settings.
2. Identify the checkpoint and dataset manifests used.
3. Preserve the run's `finetuning.json` metadata and validation history.
4. State which split produced each metric; do not use test metrics for model selection.
5. Note hardware/backend details and any changed preprocessing or decoding.

### Before sharing a change

From the repository root, run the focused checks for the files you changed. The current unit and style checks are:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
ruff check src tests scripts
python -m compileall -q src scripts tests
```

For CLI changes, also check the relevant `--help` output. For model or preprocessing changes, run parity checks when the local Paddle model and fixed cases are available. In your change summary, describe the motivation, behavior affected, checks run, results, and any limitations. Keep unrelated formatting or cleanup changes separate where practical so collaborators can review the behavioral changes clearly.

## Research notes and limitations

- The JAX recognizer accepts `[batch, 3, 48, 320]` images and produces `[batch, 40, 18,710]` CTC outputs.
- The recorded Paddle-to-JAX conversion passed parity on nine fixed and synthetic inputs, with agreement on all 360/360 timestep argmaxes. See the [parity report](docs/2026-10-17_ppocrv6-jax-parity-report.json) for details. The checkpoint files used for the run are not distributed in Git.
- A recorded fine-tuning run improved validation CER from `0.3903` to `0.1665` and raw exact match from `0.1465` to `0.4559` on 669 validation samples. Its checkpoint and run metadata were stored locally under `artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_converged`; generated checkpoint files are excluded from Git. The test split was not used for that run.
- These results describe recorded experiments, not guaranteed performance on another dataset installation, hardware setup, or recipe.
- The supplied dataset split has no writer/source groups, so the current sample-level validation metrics may be optimistic.
