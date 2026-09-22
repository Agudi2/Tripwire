# Tripwire results - test split

Target error rate: alpha = 0.02. Calibrator and threshold were fit on the calibration split only.

Expected calibration error uses 10 equal-width, right-closed bins over [0, 1]. *Before* is the extractor's best raw signal, named in the cell, because the two extractors emit different signals; *after* is the calibrated confidence.

Intervals are 95% percentile bootstrap, resampled by document, because fields within a document are correlated.

| Extractor | Docs | Fields | Precision | Recall | F1 (95% CI) | ECE before / after | Brier | Threshold | Coverage | Realized error (95% CI) | Marginal risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 374 | 1496 | 0.612 | 0.612 | 0.612 [0.590, 0.636] | 0.122 (ocr_conf) / 0.026 | 0.167 | 0.870 | 0.263 | 0.059 [0.037, 0.083] | 0.015 |

**Realized error** is the error rate among accepted fields. **Marginal risk** is the share of all fields that are accepted and wrong; that is the quantity the conformal threshold bounds by alpha = 0.02, so realized error can exceed alpha whenever coverage is below 1.
