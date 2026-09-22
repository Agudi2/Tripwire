# Tripwire failure analysis - test split, rules extractor

580 wrong fields out of 1496 (38.8%).

Causes are assigned by the first matching rule, most specific first. `unclassified` means no rule fired and the case needs eyes on it.

| Cause | Count | Share of errors |
|---|---|---|
| missing | 44 | 7.6% |
| formatting | 6 | 1.0% |
| over_capture | 12 | 2.1% |
| ocr_misread | 401 | 69.1% |
| wrong_value | 117 | 20.2% |

## By field

| Field | missing | formatting | over_capture | wrong_field | ocr_misread | hallucination | wrong_value | unclassified |
|---|---|---|---|---|---|---|---|---|
| address | 0 | 6 | 12 | 0 | 222 | 0 | 32 | 0 |
| company | 0 | 0 | 0 | 0 | 150 | 0 | 0 | 0 |
| date | 44 | 0 | 0 | 0 | 13 | 0 | 0 | 0 |
| total | 0 | 0 | 0 | 0 | 16 | 0 | 85 | 0 |

## One example per cause

| Cause | Document | Predicted | Gold |
|---|---|---|---|
| missing | sroie-00007 | `` | `16/03/2019` |
| formatting | sroie-00029 | `NO 41 LORONG MELAKA PETALING JAYA` | `NO 41 LORONG MELAKA, PETALING JAYA` |
| over_capture | sroie-00007 | `NO 30 JALAN BUKIT TINGGI, PETALING JAYA DATE: 16703/2019` | `NO 30 JALAN BUKIT TINGGI, PETALING JAYA` |
| ocr_misread | sroie-00021 | `NO 69 JALAN 8UKIT TING6I, PETALING JAYA` | `NO 69 JALAN BUKIT TINGGI, PETALING JAYA` |
| wrong_value | sroie-00024 | `NO 2 PERS1ARAN RAJA, JOHOR BAHRU DATE: 28/04/Z017` | `NO 2 PERSIARAN RAJA, JOHOR BAHRU` |
