# PP-OCRv6 Fine-Tuning Strategy with JAX

**Prepared:** 2026-10-17<br>
**Scope:** Port and fine-tune PP-OCRv6 using JAX/Flax, with RxHandBD handwritten word crops as the initial adaptation target.<br>
**Status:** Phases 0–3 have a working CTC inference port and conversion path: all 161 exported tensors map into Flax NNX, and the nine-case FP32 Paddle-vs-JAX CTC logits and probability parity gates pass at `atol=rtol=2e-4` with 100% timestep argmax agreement. CTC fine-tuning ran on all 3,794 RxHandBD training examples and stopped after three consecutive validation epochs without a significant CER improvement. Best validation was at continuation epoch 14: CER 0.1665 and raw exact match 0.4559 on 669 samples, compared with 0.3903 CER and 0.1465 exact match before fine-tuning. The best checkpoint and full run history in `finetuning.json` were stored locally under `artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_converged`; generated checkpoint files are excluded from Git. The untouched test split was not used. The objective is CTC-only because the inference artifact has no NRTR training head, and the sample-level validation split has no writer/source grouping. See [`README.md`](../README.md) for commands and evidence.

## Executive recommendation

It is feasible to fine-tune PP-OCRv6 with JAX, but there is no official JAX port or plug-in training recipe. PaddleOCR implements the model and training pipeline in PaddlePaddle. A JAX effort therefore requires reimplementing the architecture, converting the Paddle checkpoint, and demonstrating numerical parity before training. The AI Hypercomputer organization provides JAX training infrastructure examples, but no PP-OCRv6 OCR implementation. MaxText is an LLM stack; it can inform accelerator, optimizer, and checkpoint patterns, but it is not the OCR model or trainer. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [AI Hypercomputer](https://github.com/ai-hypercomputer), [MaxText](https://github.com/AI-Hypercomputer/maxtext)

For the RxHandBD use case, **begin with the recognition model only**. The current evaluation uses already-cropped handwritten words, so the detector is not part of that task. The existing held-out evaluation reports 25.92% raw exact match and 36.15% CER for pretrained PP-OCRv6 medium recognition. This is a useful baseline, but it does not establish a target score or guarantee fine-tuning will improve it. Keep the test set untouched; build train and validation sets from the dataset's training data and split by writer/source if those identifiers are available. [Current evaluation](2026-10-17_ppocrv6-rxhandbd-evaluation.md)

Use **JAX + Flax NNX + Optax + Orbax** for a custom trainer. Keep preprocessing and augmentation in a deterministic Python data pipeline initially; use Grain or a similar loader only if throughput requires it. Start on one supported accelerator and one model variant. Add multi-device sharding after the single-device model, data pipeline, checkpoint conversion, and validation loop work end to end.

## Scope and model choice

PP-OCRv6's general OCR pipeline consists of separate text detection and recognition models. Detection finds text polygons in full images; recognition transcribes cropped text lines. PaddleOCR publishes separate PP-OCRv6 configs for the two tasks. [Detection config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml), [recognition config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml)

| Goal | Model to port | Training input | Recommendation |
| --- | --- | --- | --- |
| Transcribe RxHandBD word crops | `PP-OCRv6_medium_rec` | Cropped images and transcripts | Do this first; matches the existing benchmark. |
| Find text on complete prescription pages | `PP-OCRv6_medium_det` | Page images and text polygons | Needed only if the deployed input is a full page and detector performance is insufficient. |
| Extract text from full prescription pages | Both models plus crop/ordering pipeline | Full pages, polygons, and transcripts | Treat as a later integration stage and report end-to-end metrics. |

Start with medium because it is the existing measured baseline. If accelerator memory or iteration time becomes an obstacle, compare small as a separate port target; do not silently mix variant weights or architecture settings.

## Recommended software stack

- **JAX / jaxlib:** arrays, automatic differentiation, compilation, and device execution.
- **Flax NNX:** model modules and mutable model state such as batch-normalization statistics. MaxText has migrated to NNX, making it a relevant ecosystem reference, although its production model code targets LLMs. [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- **Optax:** Adam/AdamW, schedules, gradient clipping, and optimizer transformations.
- **Orbax Checkpoint:** persist model state, optimizer state, step, and run metadata; use a checkpoint format that can be restored with the exact model/config.
- **Image/data pipeline:** Pillow or OpenCV for decode/resize, NumPy for CPU-side transforms, and an ordinary Python iterable at first. Preserve PaddleOCR's channel order, normalization, resize, padding, and augmentation semantics.
- **Optional scale-out:** JAX sharding and Grain for accelerator-host input pipelines. Keep the first run simple; the medium recognizer is roughly 19 million parameters, so a multi-host TPU system is not necessary just to fit its weights.

Pin Python and library versions in a lockfile, record accelerator type and JAX backend, and keep the Paddle reference environment separate from the JAX training environment. This makes comparison reproducible and avoids framework dependency conflicts.

## Phase 0: Freeze the baseline and data contract

1. Record the source PaddleOCR version, PP-OCRv6 model variant, model revision/hash, dictionary file, image preprocessor config, and inference decoder.
2. Retain the existing evaluation artifacts and metrics. Add per-sample predictions and error groupings by writer, image source, word length, and label category when metadata permits.
3. Confirm that RxHandBD's train and validation annotations are available. Do not use `Test_Set` or `Test_Labels.csv` for training, augmentation decisions, threshold tuning, early stopping, or dictionary changes.
4. Inspect labels for Unicode normalization, casing, punctuation, whitespace, illegible items, and transcription policy. Define a canonical transcription convention before training. Keep raw labels and any normalized training labels so transformations can be audited.
5. Split at writer or source-document level to reduce leakage from similar handwriting, repeated prescriptions, or multiple crops of the same page. If no writer metadata exists, group by source image/page if possible and document the limitation.
6. Create one manifest per split with relative image path, transcript, stable sample ID, source group, and optional quality/legibility flags. Verify each manifest's image paths and label encoding before model work.

The official PP-OCRv6 medium recognizer config uses `PPLCNetV4`, `lightsvtr` with dimensions 192 and depth 2, and a multi-head CTC/NRTR decoder. Its base training image shape is `[3, 48, 320]` with multi-scale heights 32, 48, and 64 at width 320. It uses the PP-OCRv6 character dictionary, a maximum text length of 25 in that config, and recognition augmentation. These are reproduction anchors, not assumptions that RxHandBD has the same text-length distribution or optimal scale. Inspect RxHandBD and the checkpoint's actual config before setting maximum label length or extending the vocabulary. [Official recognition config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml)

## Phase 1: Implement the JAX recognizer

Implement in separable Flax modules so each stage can be compared independently:

1. Image normalization and resize/valid-width handling.
2. Recognition-mode PPLCNetV4 stem and stages.
3. Height reduction and feature sequence construction.
4. LightSVTR local context and attention layers.
5. CTC logits and CTC loss/decode.
6. NRTR training head and loss, if matching the full official training objective.

The official architecture uses task-specific PPLCNetV4 configurations. Recognition stages include asymmetric `(2, 1)` strides to reduce image height while retaining horizontal sequence information. Implement these from source configuration rather than inferring a generic MobileNet. Match convolution padding, depthwise groups, activation functions, squeeze/excitation behavior, residual branches, normalization epsilon, and inference-time branch reparameterization. [PPLCNetV4 source](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/rec_lcnetv4.py), [PP-OCRv6 architecture](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)

Preserve the official output semantics:

- CTC logits have a time dimension, vocabulary dimension, and the same blank-token ID as the source dictionary/decoder.
- CTC input lengths must reflect the actual sequence width after convolution and any dynamic-width padding.
- Target strings must map through the same ordered dictionary as the pretrained output projection. Any new character changes output-head shape and must be handled as an intentional vocabulary migration.
- NRTR targets need exact start/end/pad token conventions and a causal target mask. If NRTR is deferred for an initial CTC-only feasibility experiment, mark that as a changed objective and compare against a Paddle-trained or Paddle-inferred reference; it is not faithful reproduction of the configured multi-loss training setup.

Prefer NHWC activations and HWIO kernels in JAX. The inspected PP-OCRv6 PIR artifact stores convolution arrays as OIHW, which must be transposed to `[H, W, I, O]`; its linear weights are `[I, O]`, matching Flax's dense kernel layout. Other Paddle checkpoint formats may differ, so inspect tensor shapes before conversion. Depthwise convolution has a separate grouped-channel mapping and should be validated explicitly. JAX convolution layout and padding are configurable; use the official semantics and parity tests rather than assuming defaults are equivalent. [JAX convolution guide](https://docs.jax.dev/en/latest/101/convolutions.html)

## Phase 2: Convert the pretrained checkpoint

### Prefer the training checkpoint

Find the matching **training** weights for the chosen PP-OCRv6 medium recognition architecture if available. A training checkpoint gives the most direct parameter mapping and may include training-specific modules. If only an inference artifact is available, it can still initialize matching inference/training parameters, but expect to create a fresh optimizer state and verify whether training-only layers—especially NRTR—have weights. PP-OCRv6 model releases include model artifacts, while PaddleOCR's training and inference artifacts serve different purposes. [PP-OCRv6 model collection](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6), [PaddleOCR model training guidance](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_train/recognition.en.md)

### Build a conversion manifest

Create a converter that loads Paddle parameters and maps every source parameter name to a Flax path. For each tensor, record source name, destination name, source/destination shapes, transform applied, and whether the tensor was loaded, initialized, or intentionally ignored. Fail conversion on unexplained missing keys or shape mismatches; do not silently skip them.

Handle at least:

- Conv and depthwise-conv axis permutations.
- Dense matrix transposes.
- BatchNorm affine scale/bias plus running mean/variance; preserve epsilon and inference behavior.
- LayerNorm parameters and any running state.
- Attention projection layout and any fused vs. separate Q/K/V projections.
- Vocabulary/output head rows and dictionary index ordering.
- Training-only auxiliary NRTR head availability.

Initialize a new optimizer after conversion. Paddle optimizer slots should not be assumed portable to Optax. Save converted parameters in a neutral, versioned artifact (for example, NPZ or safetensors plus JSON mapping metadata), then load those arrays into Flax and save the canonical training checkpoint with Orbax.

## Phase 3: Establish model parity before training

Use a small fixed set of real RxHandBD crops and synthetic shape cases, including narrow words, long words, blank/padded regions, grayscale inputs, and dimensions near resize boundaries. Run the same preprocessed tensor through Paddle and JAX. This implementation has a recorded nine-case run: four real training crops plus synthetic narrow grayscale, maximum-width, blank, resize-boundary, and tall/narrow inputs. It exercised resized widths 4–320 and a batch of `[9, 3, 48, 320]`.

Compare, in order:

1. Preprocessed tensor values and valid-width metadata.
2. Backbone output at each stage.
3. Neck output sequence.
4. CTC logits and NRTR logits.
5. Decoded text and per-sample confidence.

Use absolute/relative numerical tolerances appropriate for the selected precision, then perform strict FP32 parity before BF16/FP16 optimization. The current CTC logits and probability gates both pass `np.allclose(atol=rtol=2e-4)`. Across nine cases, logits have maximum absolute error `2.19345e-4` and mean absolute error `1.19393e-5`; probabilities have maximum absolute error `3.34799e-4`, mean absolute error `4.12256e-9`, and maximum relative error `4.20281e-4`. All 360/360 timestep argmaxes agree. The report includes the preprocessed batch SHA-256 (`e47d816b…fa1162f`) and resized widths. Layer-by-layer backbone/neck/attention comparisons were used to debug implementation, but are not saved in the parity JSON. The current artifact does not contain the NRTR training head, so NRTR parity is not claimed. Debug padding, BatchNorm state, resize rounding, dictionary IDs, and CTC blank handling before proceeding. A successful import or matching tensor shapes is not sufficient evidence of parity.

## Phase 4: Fine-tune the recognizer

### Training objective

Faithful objective: use the official CTC + NRTR multi-loss recognizer, with loss masking and sample lengths matching PaddleOCR. Run a CTC-only experiment only as an explicit lower-effort ablation if the auxiliary head or data path delays initial feasibility work.

### Initial fine-tuning recipe

Use these as conservative starting settings, then tune on validation data:

| Setting | Initial policy |
| --- | --- |
| Initialization | Converted PP-OCRv6 medium recognition weights |
| Precision | FP32 for parity; then BF16 on TPU or supported GPU after parity |
| Optimizer | Optax AdamW or Adam with explicit weight-decay mask for normalization/bias parameters |
| Learning rate | Begin below the official config's `5e-4`; sweep a small range such as `1e-5` to `1e-4` on the validation split |
| Schedule | Short warmup then cosine decay; early stopping on validation CER or normalized edit distance |
| Batch size | Choose by device memory; use gradient accumulation if needed and compare effective batch size |
| Gradient clipping | Global norm clipping to stabilize small-data updates |
| Trainable parameters | First run: full recognizer at low learning rate; follow with encoder freeze or LoRA only as a controlled comparison |
| Checkpoint selection | Best validation CER/normalized edit distance, not last epoch |
| Reproducibility | Fixed seed; log model/config hash, data manifest hash, optimizer, schedule, and step count |

The Paddle config's learning rate is not a direct prescription for a small, handwritten dataset: its batch size, data volume, augmentation, and training implementation differ. Treat it as an architecture/training reference. The Paddle fine-tuning guide also recommends mixing domain data with general examples to help preserve broad recognition ability. [Recognition config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml), [PaddleOCR fine-tuning guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_train/finetune.en.md)

### Data augmentation

Prioritize augmentations that resemble real prescription capture and handwriting variation, while preserving label readability:

- modest rotation, translation, scale, perspective, and crop/resize variation;
- brightness, contrast, blur, compression, and sensor/noise variation;
- paper background, shadow, and ink contrast changes when supported by real data;
- line/word width variation consistent with the recognizer's input policy.

Avoid aggressive elastic deformation, heavy blur, character occlusion, or synthetic distortions that change letter identity. Keep a clean, unaugmented validation set. Initially keep augmentation simpler than the general PP-OCR training recipe; add transforms only when error analysis shows a matching real-world failure mode.

### Small-data safeguards

- Use a writer/source-disjoint validation split and early stopping.
- Mix a small replay set of general printed recognition examples if available, but report RxHandBD and general-domain metrics separately.
- Track train and validation loss, CER, normalized exact match, raw exact match, and per-character confusion.
- Compare full fine-tuning with frozen-backbone/head adaptation; small data can overfit quickly.
- Do not tune the test set or use test errors to select a checkpoint.

## Phase 5: Evaluation and acceptance gates

Keep the existing evaluation definitions for comparability and add metrics useful for handwriting analysis:

- **Primary:** corpus CER and normalized exact-match accuracy, because a one-character medication/name error matters.
- **Secondary:** raw exact match, WER, per-word accuracy, edit distance by word length, and confidence calibration.
- **Diagnostic:** character confusion matrix, insertion/deletion/substitution counts, results by writer/source, and errors on numerals, abbreviations, punctuation, and case.
- **Operational:** throughput/latency on the intended accelerator and end-to-end preprocessing included or separately timed.

Proposed gates:

1. **Conversion gate:** all expected backbone/neck/CTC weights mapped; no unexplained mismatch.
2. **Parity gate:** high prediction agreement and bounded logit error between Paddle and JAX on fixed inputs before training.
3. **Training gate:** training loss decreases, validation CER improves over the pretrained JAX baseline, and no severe regression on a general-data replay set.
4. **Model selection gate:** checkpoint chosen solely on validation metrics.
5. **Holdout gate:** run the untouched test split once after decisions are locked and report metrics beside the prior baseline.

An improvement should be stated as measured on this specific split and metric. RxHandBD test performance alone cannot establish reliability for clinical or medication use; a separate, representative evaluation and human review workflow are necessary for consequential use.

## Optional Phase 6: Port detection for full-page input

Only start this after recognition fine-tuning is stable, unless the deployment input requires full-page detection. Port the official detection-mode PPLCNetV4, RepLKPAN/RepLKFPN as specified by the exact selected config, DBHead, and auxiliary heads. Implement DB target generation (shrink map, threshold map, masks), Dice/Focal loss, geometric augmentations, polygon postprocessing, and box cropping/rectification. The official medium detection config uses 640×640 training crops, Adam, and DB auxiliary supervision; verify exact source config and version before implementing because architecture/config names may evolve. [Detection config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml)

Detection training requires page-level images and polygon annotations. The current RxHandBD word-crop evaluation does not provide evidence about detector quality and cannot train or evaluate text localization. If page-level polygons are unavailable, continue with recognizer-only experiments or annotate a separate page-level subset.

For the complete pipeline, run detection, filter and rectify candidate regions, batch them into the recognizer, decode with the exact dictionary, and restore reading order. Evaluate end-to-end transcription separately from crop-level recognition so detector misses and crop errors remain distinguishable.

## Risks and mitigations

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Checkpoint conversion error | The model may train from corrupted or partly random initialization. | Explicit mapping manifest; fail-fast missing-key checks; layer-wise Paddle/JAX parity. |
| Different convolution/padding semantics | Small spatial shifts affect CTC time alignment and recognition. | Match padding/stride rules explicitly; compare intermediate tensors. |
| Dictionary or blank-token mismatch | Output indices decode to wrong characters even if logits look plausible. | Freeze dictionary hash and token IDs; test encode/decode round trips. |
| Recognition checkpoint lacks training-only head | Exact multi-loss fine-tuning may not be possible from inference weights alone. | Inspect checkpoint keys; initialize missing NRTR head explicitly and document the changed starting point, or acquire matching train weights. |
| Limited handwriting data | Fine-tuning may memorize writers and reduce generalization. | Writer/source-disjoint split, light augmentation, early stopping, general replay set. |
| Framework version drift | JAX/Flax APIs and accelerator support change. | Lock versions and pin source revisions for PaddleOCR and AI Hypercomputer references. |
| Test leakage | Inflates reported performance and harms model selection. | Keep test labels out of training and validation decisions; use a one-time final evaluation. |
| Port maintenance burden | Upstream PP-OCRv6 changes can invalidate mappings or behavior. | Pin a release and maintain a conversion/parity checklist for every upgrade. |

## Deliverable layout for an implementation

Suggested future project structure:

```text
src/ocrax/
  configs/
    rec_medium_rxhandbd.yaml
    det_medium.yaml
  src/
    ppocrv6_jax/
      backbones/lcnetv4.py
      necks/light_svtr.py
      heads/ctc.py
      heads/nrtr.py
      losses/ctc.py
      data/recognition.py
      train.py
      evaluate.py
      checkpoint_convert.py
  manifests/
    train.jsonl
    validation.jsonl
  checkpoints/
  reports/
```

Keep the JAX port isolated from the existing Paddle evaluation scripts. Record every conversion source and model/config revision in the output checkpoint metadata.

## Recommended execution order

1. Confirm RxHandBD train/validation data and writer/source grouping; freeze the current test split.
2. Port medium recognition inference in Flax and convert its checkpoint.
3. Reach the parity gate against Paddle on fixed crops.
4. Fine-tune with the official CTC + NRTR objective and a conservative small-data schedule.
5. Select a checkpoint on validation CER/normalized exact match; run final held-out evaluation.
6. Port detection only if full-page extraction is in scope and page-level polygon labels are available.
7. Add accelerator sharding or optimize kernels only after the measured workload warrants it.

If JAX/TPU execution is not a firm deployment or research requirement, use PaddleOCR's supported training stack for the model adaptation and keep JAX for components that already have a native JAX implementation. That path avoids the largest feasibility risk: reproducing Paddle's exact architecture and checkpoint behavior.

## References

- [PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv6 architecture documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
- [PP-OCRv6 medium recognition config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml)
- [PP-OCRv6 medium detection config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml)
- [PPLCNetV4 source](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/rec_lcnetv4.py)
- [PaddleOCR recognition fine-tuning guidance](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_train/finetune.en.md)
- [JAX convolution dimension and padding guide](https://docs.jax.dev/en/latest/101/convolutions.html)
- [AI Hypercomputer organization](https://github.com/ai-hypercomputer)
- [MaxText JAX training stack](https://github.com/AI-Hypercomputer/maxtext)
- [Current RxHandBD PP-OCRv6 evaluation](2026-10-17_ppocrv6-rxhandbd-evaluation.md)
