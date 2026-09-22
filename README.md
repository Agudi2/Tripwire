# Tripwire

Document extraction that flags its own mistakes. Tripwire extracts fields from receipts and forms, attaches a calibrated confidence to each one, and routes uncertain fields to human review at a controlled error rate.

> **Status:** pipeline complete and runnable end to end. The numbers below come from
> **synthetic fixtures**, not the real datasets — they verify the machinery, they are not
> benchmark results. Real SROIE/CORD/FUNSD numbers stay TBD until the raw data is present.

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

**These are synthetic-fixture numbers.** 2000 generated receipts, split 1229/397/374, isotonic
calibration, α = 0.02. They demonstrate the pipeline works; they say nothing about real-world
accuracy. Reproduce with the commands in [Run](#run).

| Extractor | Field F1 (95% CI) | ECE before / after | Coverage | Marginal risk | Error among accepted |
|---|---|---|---|---|---|
| Baseline (OCR + rules) | 0.612 [0.590, 0.636] | 0.122 → 0.026 | 0.263 | 0.015 | 0.059 |
| OCR + LLM | not run | — | — | — | — |

The LLM row is empty because the extractor is cache-first and no live run has been made; see
[Known gaps](#known-gaps).

### Reading the risk columns

Two different error rates appear above, and conflating them overstates the guarantee:

- **Marginal risk** — the share of *all* fields that are accepted and wrong. This is what the
  conformal threshold bounds by α, and it holds: 0.015 ≤ 0.02.
- **Error among accepted** — the error rate *within* the accepted set, 0.059. This is larger
  by a factor of coverage, and split-conformal does **not** bound it.

The question at the top of this README asks for the second one. Delivering it requires
Learn-then-Test or RCPS rather than the split-conformal quantile currently implemented.

### Failure analysis

`reports/errors.md` breaks every wrong field down by cause. On the fixture run:

| Cause | Share of errors |
|---|---|
| OCR misread | 69.1% |
| Wrong value picked off the page | 20.2% |
| Missing (no match found) | 7.6% |
| Over-capture (span ran too far) | 2.1% |
| Formatting only | 1.0% |

`total` accounts for nearly all the wrong-value errors — the amount rule anchors on the wrong
line in 85 of 1496 fields. That is the most actionable defect in the baseline.

## Run

Offline, on synthetic fixtures — no data download, no API key, no network:

```bash
python -m data.prepare --dataset sroie --seed 42 --fixtures --count 2000
python -m extract.run --method rules --split calibration
python -m extract.run --method rules --split test
python -m confidence.fit --split calibration --method rules --calibrator isotonic
python -m eval.report --split test --method rules
python -m eval.errors --split test --method rules
```

On real data, place the raw datasets under `data/raw/<dataset>/` and drop `--fixtures`. The
loaders never download anything and will name the directory they expected if it is absent.

The LLM extractor replays from a content-addressed cache by default. A live call happens only
with `--live` **and** `ANTHROPIC_API_KEY` in the environment; a cold cache without `--live` is
a loud error naming the missing key, so no run reaches the network by accident.

Tests:

```bash
uv run --with pytest --with numpy --with scikit-learn pytest tests/ -q
```

## Known gaps

- **No live LLM run.** The `OCR + LLM` row is unmeasured.
- **Log-probs are unavailable.** The Anthropic Messages API exposes none, so that signal is
  plumbed but never populated by a live run.
- **`agreement` may be weak.** `temperature`/`top_p` are rejected by the model, so repeated
  samples vary only by intrinsic nondeterminism. Sampling under prompt perturbations would
  restore real variance.
- **Real loaders report OCR confidence as a constant 1.0**, because the public datasets ship
  human transcriptions rather than engine scores. The `ocr_conf` signal is only meaningful
  under `--fixtures` until a real OCR engine is wired in.
- **Conformal controls marginal, not conditional, risk** — see [Reading the risk
  columns](#reading-the-risk-columns).
- **Splits are hash-based**, so ratios are approximate at small N and the publishers' official
  train/test divisions are not preserved.

## References

- Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction"
- Angelopoulos et al., "Conformal Risk Control"
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017
