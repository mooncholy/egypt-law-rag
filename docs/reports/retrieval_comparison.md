# Retrieval comparison: against `baseline` (run 1fd3e52cc5ae457f8e1533157d19152c)

Each run changes the listed params against the baseline. On 65 in-scope questions: one question moves recall@5 by 0.015, so a net change of one question is within noise.

| Run | Changed | Recall@5 | MRR | Fixed | Broken | Net |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | none | 0.569 | 0.498 | – | – | – |
| champion | rerank=True | 0.708 (+0.138) | 0.762 (+0.263) | 10 | 1 | +9 |
| rerank | rerank=True | 0.708 (+0.138) | 0.762 (+0.263) | 10 | 1 | +9 |
| rerank-rrf-k-10 | rerank=True, rrf_k=10 | 0.708 (+0.138) | 0.762 (+0.264) | 10 | 1 | +9 |
| dense-only | retrieval_mode=dense | 0.631 (+0.062) | 0.583 (+0.085) | 11 | 7 | +4 |
| rrf-k-10 | rrf_k=10 | 0.615 (+0.046) | 0.522 (+0.023) | 3 | 0 | +3 |
| cite-expansion | cite_expansion=True | 0.600 (+0.031) | 0.492 (-0.006) | 2 | 0 | +2 |
| bm25-stop-nltk | bm25_stopwords=nltk | 0.569 (+0.000) | 0.491 (-0.008) | 1 | 1 | +0 |
| repealed-heading | repealed_text=heading | 0.569 (+0.000) | 0.498 (+0.000) | 0 | 0 | +0 |
| repealed-per-range | repealed=per_range | 0.569 (+0.000) | 0.499 (+0.000) | 0 | 0 | +0 |
| rrf-k-120 | rrf_k=120 | 0.569 (+0.000) | 0.501 (+0.002) | 0 | 0 | +0 |
| bm25-stop-lucene | bm25_stopwords=lucene | 0.554 (-0.015) | 0.487 (-0.012) | 0 | 1 | -1 |

## champion (run 3b0931599c3c4b89b821eda61235c124)

- Changed: rerank=True
- Fixed at 5 (10): q008 (rule, en, english), q012 (rule, en, english), q020 (rule, ar, colloquial), q024 (rule, en, english), q030 (rule, en, english), q050 (rule_with_exception, en, english), q059 (multi_article, en, english), q060 (multi_article, ar, colloquial), q071 (cross_reference, en, english), q136 (rule_with_exception, en, english)
- Broken at 5 (1): q057 (multi_article, ar, msa)

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: cross_reference | 6 | 3 | 4 | +1 |
| kind: multi_article | 10 | 2 | 3 | +1 |
| kind: rule | 20 | 13 | 18 | +5 |
| kind: rule_with_exception | 10 | 8 | 10 | +2 |
| language: ar | 31 | 19 | 20 | +1 |
| language: en | 34 | 18 | 26 | +8 |
| register: colloquial | 9 | 2 | 4 | +2 |
| register: english | 34 | 18 | 26 | +8 |
| register: msa | 22 | 17 | 16 | -1 |

## rerank (run 0763f7010cc548bfaca16c84c59afd3b)

- Changed: rerank=True
- Fixed at 5 (10): q008 (rule, en, english), q012 (rule, en, english), q020 (rule, ar, colloquial), q024 (rule, en, english), q030 (rule, en, english), q050 (rule_with_exception, en, english), q059 (multi_article, en, english), q060 (multi_article, ar, colloquial), q071 (cross_reference, en, english), q136 (rule_with_exception, en, english)
- Broken at 5 (1): q057 (multi_article, ar, msa)

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: cross_reference | 6 | 3 | 4 | +1 |
| kind: multi_article | 10 | 2 | 3 | +1 |
| kind: rule | 20 | 13 | 18 | +5 |
| kind: rule_with_exception | 10 | 8 | 10 | +2 |
| language: ar | 31 | 19 | 20 | +1 |
| language: en | 34 | 18 | 26 | +8 |
| register: colloquial | 9 | 2 | 4 | +2 |
| register: english | 34 | 18 | 26 | +8 |
| register: msa | 22 | 17 | 16 | -1 |

## rerank-rrf-k-10 (run 68ba1b5c869d4dc0b794e7d2f1936c99)

- Changed: rerank=True, rrf_k=10
- Fixed at 5 (10): q008 (rule, en, english), q012 (rule, en, english), q020 (rule, ar, colloquial), q024 (rule, en, english), q030 (rule, en, english), q050 (rule_with_exception, en, english), q059 (multi_article, en, english), q060 (multi_article, ar, colloquial), q071 (cross_reference, en, english), q136 (rule_with_exception, en, english)
- Broken at 5 (1): q057 (multi_article, ar, msa)

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: cross_reference | 6 | 3 | 4 | +1 |
| kind: multi_article | 10 | 2 | 3 | +1 |
| kind: rule | 20 | 13 | 18 | +5 |
| kind: rule_with_exception | 10 | 8 | 10 | +2 |
| language: ar | 31 | 19 | 20 | +1 |
| language: en | 34 | 18 | 26 | +8 |
| register: colloquial | 9 | 2 | 4 | +2 |
| register: english | 34 | 18 | 26 | +8 |
| register: msa | 22 | 17 | 16 | -1 |

## dense-only (run 67cd154e6d50463e893d6de6627495dd)

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

## rrf-k-10 (run 1c19e9f1988648a2874a6319a65a0885)

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

## cite-expansion (run a56122a6931d43078811167d605a4d40)

- Changed: cite_expansion=True
- Fixed at 5 (2): q071 (cross_reference, en, english), q072 (cross_reference, ar, msa)
- Broken at 5 (0): none

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: cross_reference | 6 | 3 | 5 | +2 |
| language: ar | 31 | 19 | 20 | +1 |
| language: en | 34 | 18 | 19 | +1 |
| register: english | 34 | 18 | 19 | +1 |
| register: msa | 22 | 17 | 18 | +1 |

## bm25-stop-nltk (run bdab6ab390894927a84c0e6d55ee8fb8)

- Changed: bm25_stopwords=nltk
- Fixed at 5 (1): q050 (rule_with_exception, en, english)
- Broken at 5 (1): q134 (rule, en, english)

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: rule | 20 | 13 | 12 | -1 |
| kind: rule_with_exception | 10 | 8 | 9 | +1 |

## repealed-heading (run 1e8539c4545d413b9868d69b5a549b70)

- Changed: repealed_text=heading
- Fixed at 5 (0): none
- Broken at 5 (0): none

No group moved.

## repealed-per-range (run 3a6153c6a70d4644a7b52ed051179000)

- Changed: repealed=per_range
- Fixed at 5 (0): none
- Broken at 5 (0): none

No group moved.

## rrf-k-120 (run 1977a312a0594819aba5333096680006)

- Changed: rrf_k=120
- Fixed at 5 (0): none
- Broken at 5 (0): none

No group moved.

## bm25-stop-lucene (run fd225d3f769c4efc8e6eb5a44124d11f)

- Changed: bm25_stopwords=lucene
- Fixed at 5 (0): none
- Broken at 5 (1): q134 (rule, en, english)

| Group | Questions | Baseline found | Run found | Δ |
| --- | --- | --- | --- | --- |
| kind: rule | 20 | 13 | 12 | -1 |
| language: en | 34 | 18 | 17 | -1 |
| register: english | 34 | 18 | 17 | -1 |
