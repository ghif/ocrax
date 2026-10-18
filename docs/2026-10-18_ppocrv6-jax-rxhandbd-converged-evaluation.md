# Fine-Tuned JAX PP-OCRv6 Evaluation on RxHandBD

**Evaluation date:** 2026-10-18<br>
**Model:** Flax NNX PP-OCRv6 medium CTC recognizer fine-tuned on RxHandBD<br>
**Checkpoint:** `artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_converged` (stored locally and excluded from Git)<br>
**Dataset split:** RxHandBD test set<br>
**Prediction artifact:** `outputs/rxhandbd_ppocrv6_jax_converged_predictions.csv` (generated locally and excluded from Git)<br>
**Evaluation script:** [`src/ocrax/cli/evaluate_jax.py`](../src/ocrax/cli/evaluate_jax.py)

## Summary

The fine-tuned JAX PP-OCRv6 model transcribed **408 of 1,115 test crops exactly** (36.59% raw exact-match accuracy). Case-folding and collapsing whitespace raises exact-match accuracy to **44.93%** (501 of 1,115). Corpus character error rate (CER) is **26.20%**, and word-level accuracy, defined as `1 - WER`, is **39.69%**.

Compared with the pretrained JAX baseline evaluated on 2026-10-17, raw exact-match accuracy improved by **8.43 percentage points**, normalized exact-match accuracy by **12.28 points**, CER fell by **8.28 points**, and word-level accuracy improved by **10.75 points**. Prediction batches took **58.638 seconds**, averaging **52.59 ms per image** (**19.01 images per second**).

## Dataset and Evaluation Setup

- Dataset root: `/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD`
- Labels: `Test_Labels.csv` (`Images`, `Text` columns)
- Images: `Test_Set/`; all 1,115 word crops were evaluated in CSV row order.
- The evaluator used the JAX port's `resize_rec_image` preprocessing, Flax NNX inference, and greedy CTC decoding. No Paddle inference or training was run during evaluation.
- Runtime: `jax-cpu` Conda environment, CPU device, batch size 32, with `JAX_PLATFORMS=cpu`.
- Checkpoint fine-tuning metadata records 3,794 training samples, 669 validation samples, best continuation epoch 14, and `test_set_used: false`.

## Results

| Metric | Fine-tuned JAX PP-OCRv6 | Pretrained JAX baseline | Change |
| --- | ---: | ---: | ---: |
| Raw exact-match accuracy | 36.59% (408 / 1,115) | 28.16% (314 / 1,115) | +8.43 percentage points |
| Normalized exact-match accuracy | 44.93% (501 / 1,115) | 32.65% (364 / 1,115) | +12.28 percentage points |
| Character error rate (CER) | 26.20% | 34.48% | -8.28 percentage points |
| Word-level accuracy (`1 - WER`) | 39.69% | 28.94% | +10.75 percentage points |
| Word error rate (WER) | 60.31% | 71.06% | -10.75 percentage points |

Raw exact match requires identical case and whitespace. Normalized exact match case-folds text and collapses whitespace while retaining punctuation. CER is corpus character edit distance divided by normalized reference characters. WER is corpus word edit distance divided by normalized reference words; these rates are computed across the full test set.

## Inference Timing

| Measurement | Result |
| --- | ---: |
| Total prediction-batch time | 58.638 s |
| Average latency | 52.59 ms / image |
| Throughput | 19.01 images / second |

Timing covers image loading, resize and normalization, device transfer and synchronization, model inference, and CTC decoding for prediction batches. It excludes checkpoint restore, JAX compilation warm-up, metric calculation, and CSV writing. The pretrained baseline took 67.892 seconds under the same evaluator settings; timing is hardware and runtime dependent.

## Reproduction

From the repository root, use the `jax-cpu` Conda environment:

```bash
JAX_PLATFORMS=cpu conda run -n jax-cpu ocrax-evaluate-jax \
  --dataset-root "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD" \
  --checkpoint artifacts/checkpoints/ppocrv6_medium_rec_rxhandbd_converged \
  --batch-size 32 \
  --output outputs/rxhandbd_ppocrv6_jax_converged_predictions.csv
```

The CSV contains each filename, ground-truth text, prediction, CTC confidence, and raw correctness.
