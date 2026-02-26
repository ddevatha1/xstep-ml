# Extracted Metrics Summary

This file summarizes the numeric metrics extracted from the model evaluation graphs using OCR.

**Heatmap model**

- Accuracy: 0.9945  (from `heatmap model/heatmap model graphs/heatmap_per_class_accuracy.png`)
- Precision: 0.9946 (parsed from `heatmap model/heatmap model graphs/heatmap_summary_metrics.png` after preprocessing)
- Per-class ROC AUCs: 1.000 (9 classes) (from `heatmap model/heatmap model graphs/heatmap_roc_curves.png`)

**Ulcer (image) model**

- Accuracy: 0.8766 (from `ulcer model/ulcer model graphs/ulcer_per_class_accuracy.png`)
- Precision: 0.8785 (parsed from `ulcer model/ulcer model graphs/ulcer_summary_metrics.png` after preprocessing)
- F1-Score: 0.8766 (parsed from `ulcer model/ulcer model graphs/ulcer_summary_metrics.png` after preprocessing)
- Per-grade ROC AUCs: 0.977; 0.975; 0.968; 0.976 (from `ulcer model/ulcer model graphs/ulcer_roc_curves.png`)

Notes:

- Values were extracted using automated OCR; see `tools/ocr_extraction_results.txt` and `tools/ocr_preprocess_results.json` for full OCR outputs and raw text.
- Some metrics (e.g., Accuracy, Recall) were not consistently recognized by OCR in the summary images; consider manual verification or re-running OCR with alternate cropping if exact fidelity is required.

Preprocessed images used for OCR inspection:

- `tools/preproc_heatmap_summary_metrics.png`
- `tools/preproc_ulcer_summary_metrics.png`
