# Retrieval comparison: against `baseline` (run 9aa46ab944da432386541343d1b49189)

Each run changes the listed params against the baseline. On 65 in-scope questions: one question moves recall@5 by 0.015, so a net change of one question is within noise.

| Run | Changed | Recall@5 | MRR | Fixed | Broken | Net |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | none | 0.569 | 0.498 | – | – | – |
| dense-only | retrieval_mode=dense | 0.631 (+0.062) | 0.583 (+0.085) | 11 | 7 | +4 |
| rrf-k-10 | rrf_k=10 | 0.615 (+0.046) | 0.522 (+0.023) | 3 | 0 | +3 |
| bm25-stop-nltk | bm25_stopwords=nltk | 0.569 (+0.000) | 0.491 (-0.008) | 1 | 1 | +0 |
| repealed-per-range | repealed=per_range | 0.569 (+0.000) | 0.499 (+0.000) | 0 | 0 | +0 |
| rrf-k-120 | rrf_k=120 | 0.569 (+0.000) | 0.501 (+0.002) | 0 | 0 | +0 |
| bm25-stop-lucene | bm25_stopwords=lucene | 0.554 (-0.015) | 0.487 (-0.012) | 0 | 1 | -1 |

## dense-only (run f4aa8a66706c41f09fbe7be01d8b49f0)

- Changed: retrieval_mode=dense
- Fixed at 5 (11): q001 (rule, en, english), q008 (rule, en, english), q012 (rule, en, english), q020 (rule, ar, colloquial), q050 (rule_with_exception, en, english), q060 (multi_article, ar, colloquial), q071 (cross_reference, en, english), q100 (one_language_only, ar, colloquial), q101 (lay_term, en, english), q136 (rule_with_exception, en, english), q140 (cross_reference, en, english)
- Broken at 5 (7): q016 (rule, en, english), q035 (rule_with_exception, ar, msa), q047 (rule_with_exception, ar, msa), q057 (multi_article, ar, msa), q075 (cross_reference, en, english), q106 (lay_term, ar, colloquial), q142 (one_language_only, en, english)

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: cross_reference | 6 | 3 | 4 | +1 |
| kind: rule | 20 | 13 | 16 | +3 |
| language: ar | 31 | 19 | 18 | -1 |
| language: en | 34 | 18 | 23 | +5 |
| register: colloquial | 9 | 2 | 4 | +2 |
| register: english | 34 | 18 | 23 | +5 |
| register: msa | 22 | 17 | 14 | -3 |

## rrf-k-10 (run 9ddd95d58514469282d119df5c268d8f)

- Changed: rrf_k=10
- Fixed at 5 (3): q008 (rule, en, english), q020 (rule, ar, colloquial), q050 (rule_with_exception, en, english)
- Broken at 5 (0): none

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: rule | 20 | 13 | 15 | +2 |
| kind: rule_with_exception | 10 | 8 | 9 | +1 |
| language: ar | 31 | 19 | 20 | +1 |
| language: en | 34 | 18 | 20 | +2 |
| register: colloquial | 9 | 2 | 3 | +1 |
| register: english | 34 | 18 | 20 | +2 |

## bm25-stop-nltk (run 3cdd202e376a41e58d6e0f551cbcd1ee)

- Changed: bm25_stopwords=nltk
- Fixed at 5 (1): q050 (rule_with_exception, en, english)
- Broken at 5 (1): q134 (rule, en, english)

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: rule | 20 | 13 | 12 | -1 |
| kind: rule_with_exception | 10 | 8 | 9 | +1 |

## repealed-per-range (run bd10a6a1f5ad45308b903193027b45f3)

- Changed: repealed=per_range
- Fixed at 5 (0): none
- Broken at 5 (0): none

No group moved.

## rrf-k-120 (run 7662dc9b8d75472da76e69fb4c42c3af)

- Changed: rrf_k=120
- Fixed at 5 (0): none
- Broken at 5 (0): none

No group moved.

## bm25-stop-lucene (run a17cb14f74c14fb396c5e9a748c64582)

- Changed: bm25_stopwords=lucene
- Fixed at 5 (0): none
- Broken at 5 (1): q134 (rule, en, english)

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: rule | 20 | 13 | 12 | -1 |
| language: en | 34 | 18 | 17 | -1 |
| register: english | 34 | 18 | 17 | -1 |
