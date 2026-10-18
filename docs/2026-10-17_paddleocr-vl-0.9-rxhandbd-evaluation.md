# PaddleOCR-VL-0.9B Evaluation on RxHandBD

**Evaluation date:** 2026-10-17<br>
**Model:** `PaddleOCR-VL-0.9B` (pretrained, no fine-tuning)<br>
**Dataset split:** RxHandBD test set<br>
**Prediction artifact:** `outputs/rxhandbd_paddleocr_vl_0_9_predictions.csv` (generated locally and excluded from Git)<br>
**Evaluation script:** [`src/ocrax/cli/evaluate_vl.py`](../src/ocrax/cli/evaluate_vl.py)

## Summary

PaddleOCR-VL-0.9B transcribed **265 of 1,115 crops exactly** (23.77% raw exact-match accuracy). Case-folding and collapsing whitespace raises exact-match accuracy to **27.80%**. Corpus character error rate (CER) is **42.44%**, and word-level accuracy, defined as `1 - WER`, is **18.27%**.

On CPU with one crop per prediction call, measured prediction calls took **2,243.010 seconds** total, averaging **2,011.67 ms per image** and **0.50 images per second**. For quality, PP-OCRv6 performed better on all four reported recognition metrics in the existing [PP-OCRv6 evaluation](2026-10-17_ppocrv6-rxhandbd-evaluation.md). The latency figures are hardware and configuration dependent and should not be treated as a controlled speed comparison.

## Dataset and Evaluation Setup

- Dataset root: `/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD`
- Labels: `Test_Labels.csv` (`Images`, `Text` columns)
- Images: `Test_Set/`; all 1,115 word crops were evaluated in CSV row order.
- The model was run directly on each word crop with the OCR prompt. Layout detection, document orientation classification, and document unwarping were disabled. No detection, fine-tuning, medication dictionary, or language-model correction was applied.
- Runtime: Python 3.13.15, PaddleOCR 3.7.0, PaddleX 3.7.2, PaddlePaddle 3.3.1; CPU device; batch size 1.
- The script checked the local checkpoint files before pipeline construction and passed the checkpoint directory explicitly. The run loaded weights from `/Users/mghifary/.paddlex/official_models/PaddleOCR-VL-0.9B`; it did not use a model lookup or download fallback.
- Generation was capped at 64 new tokens per word crop to bound unusually long decodes. The same cap was applied to every crop.

## PaddleOCR-VL Model Configuration

| Setting | Value |
| --- | --- |
| Model name | `PaddleOCR-VL-0.9B` |
| Approximate parameter class | 0.9 billion parameters, as indicated by the model variant name |
| PaddleOCR API | `PaddleOCRVL` |
| Pipeline version | `v1` (PaddleOCR-VL) |
| Recognition backend | Native Paddle |
| Checkpoint directory | `/Users/mghifary/.paddlex/official_models/PaddleOCR-VL-0.9B` (local cache) |
| Device | `cpu` |
| Inference batch size | 1 image per call |
| Maximum generated tokens | 64 per crop |
| Text prompt | `ocr` |
| Language and vision model | Checkpoint configuration identifies `PaddleOCRVLForConditionalGeneration`; language model has 18 layers, hidden size 1,024, 16 attention heads and 2 key/value heads. Vision configuration has 27 layers, hidden size 1,152, 16 attention heads and 384-pixel image size. |
| Layout detection | Disabled |
| Document orientation classification | Disabled |
| Document unwarping | Disabled |
| Confidence output | Not provided in a comparable form by this API; the CSV confidence field is blank. |

The architecture values above come from the cached checkpoint's `config.json`. The “0.9B” parameter count is reported as the model variant's approximate size; it was not independently recounted from tensors for this evaluation.

## Results

| Metric | PaddleOCR-VL-0.9B | PP-OCRv6 medium | Definition |
| --- | ---: | ---: | --- |
| Raw exact-match accuracy | 23.77% (265 / 1,115) | 25.92% (289 / 1,115) | Exact string match including case and whitespace. |
| Normalized exact-match accuracy | 27.80% (310 / 1,115) | 30.04% (335 / 1,115) | Exact match after case-folding and collapsing whitespace; punctuation is retained. |
| Character error rate (CER) | 42.44% | 36.15% | Corpus character edit distance divided by normalized reference characters. |
| Word-level accuracy | 18.27% | 27.00% | `1 - WER`, with whitespace-delimited tokens after case-folding and whitespace normalization. |
| Word error rate (WER) | 81.73% | 73.00% | Corpus word edit distance divided by normalized reference words. |

For PaddleOCR-VL-0.9B, CER is 3,109 character edits over 7,325 normalized reference characters. WER is 1,011 word edits over 1,237 normalized reference words. The corpus error rates are computed over the complete test set rather than averaged per image.

PaddleOCR-VL-0.9B trails PP-OCRv6 by 2.15 percentage points in raw exact match, 2.24 points in normalized exact match, and 8.73 points in word-level accuracy. Its CER is 6.29 points higher. These are direct comparisons on the same split with the same metric normalization rules.

## Inference Timing

| Measurement | Result |
| --- | ---: |
| Total prediction-call time | 2,243.010 s (37 min 23.010 s) |
| Average latency | 2,011.67 ms / image |
| Throughput | 0.50 images / second |

Timing is the sum of wall time around each `PaddleOCRVL.predict()` call, including materializing its returned results. It excludes model initialization, checkpoint loading, metric calculation, and CSV writing. A maximum of 64 generated tokens was supplied on each call. PP-OCRv6 was run on CPU at batch size 32, while this model was run on CPU at batch size 1; runtime versions and generation behavior also differ. Therefore these latency measurements describe their respective runs and are not a like-for-like throughput benchmark.

## Interpretation and Limitations

- This is a word-crop recognition benchmark, so the evaluation isolates recognition from page layout and text detection.
- The pretrained model makes substantial character and word errors on this handwritten prescription dataset. These results do not support directly using its output as medication instructions.
- Normalized exact match ignores case and whitespace differences, while retaining punctuation. CER and WER use the same normalization for references and predictions.
- The 64-token generation limit bounds decoding time for each word crop and can truncate abnormally long generations. It was applied uniformly and is part of this run's configuration.
- The confidence column is empty because PaddleOCR-VL does not expose a comparable per-crop confidence score in this pipeline result.
- This is one held-out split and one inference configuration. It does not measure performance by medicine class, handwriting author, or prescription source, and it does not establish clinical safety.

## Reproduction

Use an environment with Python 3.13.x, PaddleOCR 3.7.0, PaddleX 3.7.2 and PaddlePaddle 3.3.1, including PaddleX's OCR dependencies (`paddlex[ocr]`). The checkpoint must already be present locally; the evaluator checks for it and never downloads it.

```bash
ocrax-evaluate-vl \
  --dataset-root "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD" \
  --model-dir "/Users/mghifary/.paddlex/official_models/PaddleOCR-VL-0.9B" \
  --device cpu \
  --output outputs/rxhandbd_paddleocr_vl_0_9_predictions.csv
```

The script prints raw and normalized exact-match accuracy, CER, word-level accuracy, WER, prediction-call time, average latency, and throughput. It writes the original ground-truth and predicted strings plus raw correctness for every image to the prediction CSV.
