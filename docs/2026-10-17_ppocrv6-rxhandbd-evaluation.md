# PP-OCRv6 Evaluation on RxHandBD

**Evaluation date:** 2026-10-17<br>
**Model:** `PP-OCRv6_medium_rec` (pretrained, no fine-tuning)<br>
**Dataset split:** RxHandBD test set<br>
**Prediction artifact:** `outputs/rxhandbd_ppocrv6_predictions.csv` (generated locally and excluded from Git)<br>
**Evaluation script:** [`src/ocrax/cli/evaluate_paddle.py`](../src/ocrax/cli/evaluate_paddle.py)

## Summary

The pretrained PP-OCRv6 medium recognizer transcribed **289 of 1,115 crops exactly** (25.92% raw exact-match accuracy). Case-folding and collapsing whitespace raises exact-match accuracy to **30.04%**. Corpus character error rate (CER) is **36.15%**, and word-level accuracy, defined as `1 - WER`, is **27.00%**.

On CPU, the measured prediction calls took **82.648 seconds** total, averaging **74.12 ms per image** and **13.49 images per second**. This is a useful baseline, but the recognition error is too high for directly using the output as medication instructions.

## Dataset and Evaluation Setup

- Dataset root: `/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD`
- Labels: `Test_Labels.csv` (`Images`, `Text` columns)
- Images: `Test_Set/`; 1,115 labeled word crops were evaluated in CSV row order.
- The evaluation uses the pretrained `PP-OCRv6_medium_rec` text-recognition model directly on each crop. It does not run text detection, fine-tune the model, or apply a medication dictionary or language-model correction.
- Runtime: Python 3.11.11, PaddleOCR 3.7.0, PaddlePaddle 3.3.1; CPU device; batch size 32.

## PP-OCRv6 Model Configuration

### Configuration Used for This Run

| Setting | Value |
| --- | --- |
| PaddleOCR model name | `PP-OCRv6_medium_rec` |
| Parameter count | 19.2 million parameters (recognition checkpoint) |
| PaddleOCR API | `TextRecognition` recognition module |
| Weights | PaddleOCR pretrained inference checkpoint; no dataset fine-tuning |
| Device | `cpu` |
| Inference batch size | 32 images |
| Engine | PaddleOCR local default (`paddle_static`) |
| Image decoding | BGR image input |
| Recognition input tensor shape | `3 × 48 × 320` (channels × height × width), resized by the model's recognition preprocessor |
| Output decoding | `CTCLabelDecode` |
| Detection / orientation / document correction | Not used; input items are already cropped word images |

The installed checkpoint metadata (`PP-OCRv6_medium_rec/inference.yml`) specifies the BGR decode, `3 × 48 × 320` recognition shape, and CTC output decoder. PaddleOCR's model documentation describes PP-OCRv6 recognition as PPLCNetV4 + LightSVTR with CTC/NRTR multi-head decoding; it states that PP-OCRv6 medium/small recognition supports 50 languages. The specific `PP-OCRv6_medium_rec` model card reports 19.2 million parameters; this is the recognizer count, not the combined detector-plus-recognizer pipeline. The architecture and language coverage are model-family documentation, while the table above records the inference settings used here. See the [official text-recognition documentation](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html) and [PP-OCRv6 medium recognizer model card](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec).

## Results

| Metric | Result | Definition |
| --- | ---: | --- |
| Raw exact-match accuracy | 25.92% (289 / 1,115) | Predicted string exactly equals the label, including case and whitespace. |
| Normalized exact-match accuracy | 30.04% (335 / 1,115) | Exact match after case-folding and collapsing whitespace. Punctuation is retained. |
| Character error rate (CER) | 36.15% | Corpus character edit distance divided by reference characters after case-folding and whitespace normalization. |
| Word-level accuracy | 27.00% | `1 - WER`; WER uses whitespace-delimited tokens and word-level Levenshtein distance after case-folding and whitespace normalization. |
| Word error rate (WER) | 73.00% | 903 word edits / 1,237 reference words, counting substitutions, deletions, and insertions. |
| Total prediction-call time | 82.648 s | Sum of wall time around all PaddleOCR `predict()` calls, including result iteration. |
| Average inference latency | 74.12 ms / image | Total prediction-call time divided by 1,115 images. |
| Inference throughput | 13.49 images / second | 1,115 images divided by total prediction-call time. |

CER comprises 2,648 character edits over 7,325 normalized reference characters. These corpus-level error rates are calculated across the full test set rather than averaged per image.

## Interpretation and Limitations

- This dataset is a word-crop recognition benchmark, so the evaluation isolates recognition from page layout and text detection.
- A 27.00% word-level accuracy means most reference tokens are not reproduced correctly under the stated WER definition. The result supports the earlier assessment that this pretrained model is a baseline, not a reliable prescription transcription solution without domain adaptation and human review.
- Normalized exact-match accuracy is higher than raw exact-match accuracy because case and whitespace differences are ignored; punctuation remains significant.
- The reported latency is for CPU inference at batch size 32. It excludes model initialization or download, dataset validation, metric computation, and CSV writing. Other devices, batch sizes, hardware, and runtime conditions may produce different timings.
- This is one held-out split and one model configuration. It does not measure performance by medicine class, handwriting author, or prescription source, and it does not establish clinical safety.
- Confidence scores are emitted for each prediction in the CSV, but this run does not calibrate them or select a confidence threshold.

## Reproduction

From the repository root, run:

```bash
ocrax-evaluate \
  --dataset-root "/Users/mghifary/Work/Code/AI/data/18478741/RxHandBD" \
  --device cpu \
  --batch-size 32 \
  --output outputs/rxhandbd_ppocrv6_predictions.csv
```

The script requires PaddleOCR 3.7 or newer. On first use, PaddleOCR may download the pretrained model. It prints the accuracy, CER, word-level accuracy, prediction-call time, average latency, and throughput; it writes per-image labels, predictions, confidence scores, and raw correctness to the prediction CSV.
