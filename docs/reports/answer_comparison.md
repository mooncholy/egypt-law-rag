# Answer comparison

| Run | faithfulness | answer_relevancy | context_precision | context_recall | Answered | Gate refused | Generator refused | ...with the expected articles given |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| answers-champion | 0.576 | 0.653 | 0.824 | 0.755 | 51 | 14 | 11 | 8 |
| answers-baseline | 0.550 | 0.622 | 0.599 | 0.504 | 65 | 0 | 22 | 10 |
| answers-dense-only | 0.586 | 0.709 | 0.589 | 0.585 | 65 | 0 | 16 | 11 |
| answers-rrf-k-10 | 0.524 | 0.611 | 0.561 | 0.590 | 65 | 0 | 20 | 10 |
| answers-semantic | 0.616 | 0.623 | 0.543 | 0.515 | 65 | 0 | 21 | 9 |
| answers-champion-repeat | 0.566 | 0.652 | 0.826 | 0.755 | 51 | 14 | 11 | 8 |
| answers-prompt-v3 | 0.408 | 0.551 | 0.784 | 0.774 | 51 | 14 | 20 | 18 |
| answers-v3config-3b | 0.693 | 0.655 | 0.838 | 0.819 | 54 | 11 | 9 | 9 |
| answers-7b-prompt-v2 | 0.868 | 0.703 | 0.834 | 0.780 | 57 | 8 | 3 | 2 |

## Where faithfulness is lost

Points lost are 1 − faithfulness, summed over a run's scored answers; each answer's loss is blamed on one stage (`loss_stage`). Gate refusals aren't scored, so a run that refuses more loses fewer points.

| Run | Scored | Points lost | Retrieval | Generator refused | Generator answered |
| --- | --- | --- | --- | --- | --- |
| answers-champion | 51 | 21.6 | 4.2 (20%) | 8.0 (37%) | 9.4 (43%) |
| answers-baseline | 65 | 29.2 | 13.1 (45%) | 7.9 (27%) | 8.2 (28%) |
| answers-dense-only | 65 | 26.9 | 9.1 (34%) | 8.8 (33%) | 8.9 (33%) |
| answers-rrf-k-10 | 64 | 30.5 | 11.9 (39%) | 7.8 (26%) | 10.8 (35%) |
| answers-semantic | 65 | 25.0 | 7.9 (32%) | 6.7 (27%) | 10.3 (41%) |
| answers-champion-repeat | 51 | 22.1 | 4.6 (21%) | 7.0 (32%) | 10.5 (47%) |
| answers-prompt-v3 | 51 | 30.2 | 5.7 (19%) | 17.0 (56%) | 7.5 (25%) |
| answers-v3config-3b | 54 | 16.6 | 2.3 (14%) | 7.3 (44%) | 7.0 (42%) |
| answers-7b-prompt-v2 | 57 | 7.5 | 2.3 (30%) | 1.2 (17%) | 4.0 (53%) |

## On the 24 questions every run answered with substance

| Run | faithfulness | faithfulness (ar) | faithfulness (en) | context_precision | context_recall |
| --- | --- | --- | --- | --- | --- |
| answers-champion | 0.643 | 0.540 | 0.849 | 0.744 | 0.785 |
| answers-baseline | 0.697 | 0.609 | 0.874 | 0.726 | 0.604 |
| answers-dense-only | 0.706 | 0.600 | 0.918 | 0.628 | 0.646 |
| answers-rrf-k-10 | 0.634 | 0.491 | 0.921 | 0.682 | 0.694 |
| answers-semantic | 0.682 | 0.575 | 0.895 | 0.659 | 0.646 |
| answers-champion-repeat | 0.628 | 0.526 | 0.833 | 0.728 | 0.771 |
| answers-prompt-v3 | 0.751 | 0.657 | 0.938 | 0.680 | 0.799 |
| answers-v3config-3b | 0.739 | 0.644 | 0.931 | 0.779 | 0.830 |
| answers-7b-prompt-v2 | 0.902 | 0.880 | 0.945 | 0.776 | 0.801 |

## Repeatability (C20): answers-champion against answers-champion-repeat

Faithfulness 0.5759 against 0.5663: a difference of 0.0096 (C20 allows 0.02). 65 of 65 answers are word-for-word identical, so the difference is the judge's.
