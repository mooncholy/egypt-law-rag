# Answer comparison

| Run | faithfulness | answer_relevancy | context_precision | context_recall | Answered | Gate refused | Generator refused | ...with the expected articles given |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answers-champion | 0.576 | 0.653 | 0.824 | 0.755 | 51 | 14 | 14 | 11 |
| answers-baseline | 0.550 | 0.622 | 0.599 | 0.504 | 65 | 0 | 22 | 10 |
| answers-dense-only | 0.586 | 0.709 | 0.589 | 0.585 | 65 | 0 | 20 | 14 |
| answers-rrf-k-10 | 0.524 | 0.611 | 0.561 | 0.590 | 65 | 0 | 24 | 14 |
| answers-semantic | 0.616 | 0.623 | 0.543 | 0.515 | 65 | 0 | 22 | 9 |
| answers-champion-repeat | 0.566 | 0.652 | 0.826 | 0.755 | 51 | 14 | 14 | 11 |

## On the 25 questions every run answered with substance

| Run | faithfulness | faithfulness (ar) | faithfulness (en) | context_precision | context_recall |
| --- | --- | --- | --- | --- | --- |
| answers-champion | 0.669 | 0.540 | 0.897 | 0.756 | 0.793 |
| answers-baseline | 0.714 | 0.609 | 0.903 | 0.726 | 0.620 |
| answers-dense-only | 0.678 | 0.600 | 0.816 | 0.623 | 0.660 |
| answers-rrf-k-10 | 0.625 | 0.491 | 0.863 | 0.701 | 0.707 |
| answers-semantic | 0.698 | 0.575 | 0.915 | 0.660 | 0.660 |
| answers-champion-repeat | 0.657 | 0.526 | 0.889 | 0.743 | 0.780 |

## Repeatability (C20): answers-champion against answers-champion-repeat

Faithfulness 0.5759 against 0.5663: a difference of 0.0096 (C20 allows 0.02). 65 of 65 answers are word-for-word identical, so the difference is the judge's.
