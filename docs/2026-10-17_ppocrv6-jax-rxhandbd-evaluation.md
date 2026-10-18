# JAX PP-OCRv6 Evaluation on RxHandBD

**Evaluation date:** 2026-10-17<br>
**Model:** Flax NNX port of `PP-OCRv6_medium_rec` (converted pretrained inference weights; no fine-tuning)<br>
**Dataset split:** RxHandBD test set<br>
**Prediction artifact:** `outputs/rxhandbd_ppocrv6_jax_predictions.csv` (generated locally and excluded from Git)<br>
**Evaluation script:** [`src/ocrax/cli/evaluate_jax.py`](../src/ocrax/cli/evaluate_jax.py)<br>
**JAX port:** [`src/ocrax/`](../src/ocrax/)

## Summary

The JAX PP-OCRv6 medium port transcribed **314 of 1,115 crops exactly** (28.16% raw exact-match accuracy). Case-folding and collapsing whitespace raises exact-match accuracy to **32.65%**. Corpus character error rate (CER) is **34.48%**, and word-level accuracy, defined as `1 - WER`, is **28.94%**.

After JAX compilation warm-up, the measured CPU prediction batches took **67.892 seconds**, averaging **60.89 ms per image** and **16.42 images per second**. The scores are similar to the existing Paddle PP-OCRv6 results, but the end-to-end prediction strings differ on many crops because the two evaluation paths use different image-resizing implementations. See the comparison and preprocessing notes below.

## Dataset and Evaluation Setup

- Dataset root: `/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD`
- Labels: `Test_Labels.csv` (`Images`, `Text` columns)
- Images: `Test_Set/`; all 1,115 word crops were evaluated in CSV row order.
- The script loads the converted Orbax checkpoint at `artifacts/checkpoints/ppocrv6_medium_rec_orbax`, applies the JAX port's `resize_rec_image` preprocessing, runs the Flax NNX model, and greedily decodes the CTC output. It does not invoke Paddle inference or fine-tune the model.
- Runtime: Python 3.13.14, JAX 0.11.1, Flax 0.12.9, Orbax Checkpoint 0.12.4, NumPy 2.5.2; CPU device; batch size 32.
- `JAX_PLATFORMS=cpu` is set to avoid selecting an unavailable experimental Metal backend on this host.

## JAX Model Configuration

| Setting | Value |
| --- | --- |
| Model | `PP_OCRv6MediumRec` from `ocrax.infrastructure.models.ppocrv6` |
| Architecture | PPLCNetV4 recognition backbone, LightSVTR neck, linear CTC projection |
| Parameters | About 19.2 million, as recorded in the JAX port README |
| Weights | Converted Orbax checkpoint from the local Paddle inference graph; no fine-tuning |
| Checkpoint path | `artifacts/checkpoints/ppocrv6_medium_rec_orbax` (about 66 MB) |
| Input | BGR, `3 × 48 × 320`, float32 normalized to `[-1, 1]` |
| Output | 40 time steps × 18,710 CTC classes |
| Decoder | Greedy CTC; ID 0 is blank, repeated adjacent IDs are collapsed, and the final dictionary token is space |
| Batch size | 32 images |
| Device | CPU |
| Confidence | Mean top-1 probability over decoded nonblank characters, matching the Paddle CTC decoder's scoring rule |

The checkpoint's conversion metadata records that all 161 source tensors were consumed. The port README reports parity on nine fixed and synthetic inputs: logits and probabilities pass `atol=rtol=2e-4`, with all 360 timestep argmaxes agreeing. That check validates inference for identical preprocessed tensors; it does not guarantee that the JAX and Paddle application preprocessors construct identical tensors from image files.

## Results and Paddle Baseline

| Metric | JAX PP-OCRv6 | Paddle PP-OCRv6 baseline | Definition |
| --- | ---: | ---: | --- |
| Raw exact-match accuracy | 28.16% (314 / 1,115) | 25.92% (289 / 1,115) | Exact string match including case and whitespace. |
| Normalized exact-match accuracy | 32.65% (364 / 1,115) | 30.04% (335 / 1,115) | Exact match after case-folding and collapsing whitespace; punctuation is retained. |
| Character error rate (CER) | 34.48% | 36.15% | Corpus character edit distance divided by normalized reference characters. |
| Word-level accuracy | 28.94% | 27.00% | `1 - WER`, over whitespace-delimited tokens after case-folding and whitespace normalization. |
| Word error rate (WER) | 71.06% | 73.00% | Corpus word edit distance divided by normalized reference words. |

For JAX, CER is 2,526 character edits over 7,325 normalized reference characters. WER is 879 word edits over 1,237 normalized reference words. The corpus error rates are calculated across the complete test set rather than averaged per image.

### Preprocessing Comparability

Both evaluations use the same labels, images, row order, and metric functions, but their image preprocessing is not pixel-identical. The JAX port's `resize_rec_image` uses Pillow bilinear resizing and BGR channel order. PaddleOCR's `TextRecognition` wrapper uses PaddleX's OpenCV-based resize path and RGB image reading. The JAX port's published parity check feeds an identical tensor to Paddle and JAX; it does not compare the full image-file-to-text pipelines on this test set.

The output strings differ on 702 of 1,115 images. Because handwriting recognition can change when resizing and channel conversion change pixel values, the metric differences above should be read as end-to-end results for these two pipelines, not as a controlled comparison of Paddle versus JAX arithmetic. A strictly controlled backend comparison would feed the same preprocessed arrays to both implementations for all 1,115 images.

## Inference Timing

| Measurement | Result |
| --- | ---: |
| Total prediction-batch time | 67.892 s |
| Average latency | 60.89 ms / image |
| Throughput | 16.42 images / second |

Timing sums wall time around each batch after compiling the full and remainder batch shapes. It includes image loading, the JAX port's resize/normalization, device transfer and synchronization, model inference, and CTC decoding. It excludes checkpoint restore, compilation warm-up, metric calculation, and CSV writing. The earlier Paddle baseline reports 82.648 seconds, 74.12 ms per image, and 13.49 images per second at batch size 32. These figures are indicative only: the timing boundaries, preprocessors, and runtimes differ.

## Interpretation and Limitations

- This is a word-crop recognition benchmark, so it isolates recognition from page layout and text detection.
- The JAX result is a baseline evaluation of the converted inference checkpoint; it is not a fine-tuned RxHandBD model and does not validate a training implementation.
- The JAX port README notes that the exported inference graph contains the CTC branch only; the training-only NRTR auxiliary decoder weights are not included.
- JAX and Paddle inference are numerically close when they receive the same preprocessed tensor, as documented in the port parity report. The end-to-end test scores differ partly because this evaluation currently uses the JAX port's Pillow preprocessing while the Paddle baseline uses PaddleX/OpenCV preprocessing.
- The result does not measure performance by medicine class, handwriting author, or prescription source, and does not establish clinical safety.

## Reproduction

Use the `jax-cpu` Conda environment or an environment containing JAX, Flax NNX, Orbax Checkpoint, NumPy, Pillow, and PyYAML. From the repository root, run:

```bash
JAX_PLATFORMS=cpu conda run -n jax-cpu ocrax-evaluate-jax \
  --dataset-root "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD" \
  --checkpoint artifacts/checkpoints/ppocrv6_medium_rec_orbax \
  --batch-size 32 \
  --output outputs/rxhandbd_ppocrv6_jax_predictions.csv
```

The script prints raw and normalized exact-match accuracy, CER, word-level accuracy, WER, prediction-batch time, average latency, and throughput. The per-image CSV contains filenames, original ground truth and predictions, CTC confidence, and raw correctness.
