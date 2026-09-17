# Tripwire

Document extraction that flags its own mistakes. Tripwire extracts fields from receipts and forms, attaches a calibrated confidence to each one, and routes uncertain fields to human review at a controlled error rate.

> **Status:** in progress. Results are TBD until measured on the held-out test split.

## Question

How many fields can we auto-accept while keeping the error rate on accepted fields below a target? Tripwire answers with calibration and conformal prediction, not a hand-picked threshold.

## Data

- **SROIE:** receipts (company, date, address, total)
- **CORD:** receipts with line items
- **FUNSD:** noisy scanned forms

Fixed train / calibration / test splits. Test is used only for final numbers.

## Method

- **Extractors:** OCR + rules baseline, OCR + LLM with a JSON schema. LLM outputs are cached for reproducibility.
- **Confidence signals:** agreement across sampled outputs, OCR confidence, format validators, token log-probs where available.
- **Calibration:** Platt or isotonic, fit on the calibration split.
- **Threshold:** conformal risk control picks the acceptance cutoff for a target error rate.

## Metrics

Field-level F1, expected calibration error, Brier score, risk vs coverage curve, and realized error at target. Headline numbers include 95% bootstrap CIs resampled by document.

## Results

| Extractor | Field F1 | ECE before / after | Coverage at 2% error |
|---|---|---|---|
| Baseline | TBD | TBD | TBD |
| OCR + LLM | TBD | TBD | TBD |

Failure analysis by cause (OCR misread, wrong field, formatting, hallucination) is in `reports/errors.md`.

## Run

```bash
python -m data.prepare --dataset sroie --seed 42
python -m extract.run --method llm
python -m confidence.fit --split calibration
python -m eval.report --split test
```

## References

- Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction"
- Angelopoulos et al., "Conformal Risk Control"
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017
