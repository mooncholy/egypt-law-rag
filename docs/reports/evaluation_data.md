# Evaluation data and its limits

Two reference sets check the system, and each answers a different question:

- **Gold sample** (`data/gold/articles_gold.json`): 20 articles transcribed from the printed pages (the handbook's spot-check size). It asks *"Is our digital text faithful to the print?"* The `validate` stage compares the corpus against it.
- **Retrieval eval set** (`data/gold/retrieval_eval.jsonl`): 142 questions (68 Arabic, 20 of them colloquial; 74 English), each with the articles that govern the answer. It asks *"Does a question find the right provisions?"* The question types include rules with exceptions, cross-references, repealed ranges, passages printed in one language only, and 11 questions the Code doesn't answer.
  - **Bilingual pairs:** 20 of the questions (q123 to q142) are translations of earlier ones, so the same question can be asked in both languages and the two retrievals compared. Each twin and its source share a `pair_id` (p01 to p20), and `translated_from` names the source. Ten go from English into Modern Standard Arabic and ten from Arabic (four of them colloquial) into English. They cover every in-scope kind, roughly in proportion; out-of-scope questions are left out, since they have no article to agree on. A twin keeps its source's articles, `match`, `legal_basis` and `difficulty`. Like the rest of the set, no twin shares five consecutive words with its governing articles (checked on folded tokens).

## Truth point: who made them

Neither set was made by a legal professional. Both were produced by an LLM (Claude Opus 5.5) under time constraints, and the owner hasn't spot-checked them yet. In addition, the same model also helped build the pipeline the sets are meant to test. Read every number computed from them with that in mind:

- **The gold sample is evidence, not proof.** The articles were transcribed from page images, never from the PDF's text layer, and before looking at the corpus text. That keeps the transcription independent of the extraction bugs. It doesn't make it independent of the model's own reading errors. Twenty articles can show that a defect class exists. They can't show its absence: with no defect in 20 articles, the per-article defect rate could still be as high as 15% (the rule of three, 3/n).
- **The eval labels are one annotator's legal judgment.** Each cited article was checked against the article's text. Whether it is the provision a lawyer would cite is still a judgment, and no second annotator checked it.
- **The pairs test translation, not two native askers.** Each twin was translated by the same model from its source, so it carries the source's framing and difficulty. A colloquial source loses its dialect in English, and an English source becomes formal Arabic. Agreement within a pair measures whether retrieval is consistent across languages for one phrasing, not whether Arabic and English speakers would be served equally.
- **The questions may be easier than real ones.** They were written by a model that had read the article index, so they lean toward the Code's vocabulary. No question shares five consecutive words with its article, but shorter overlaps remain. Retrieval scores on this set are likely an upper bound on what real users will see.

## What would have been done differently

- **Gold set:** about 30 articles chosen to cover every hard case (page breaks, errata, highlighted text, repealed ranges, numbered paragraphs). Two people fluent in legal Arabic would transcribe them independently from the print, then resolve disagreements against the page. Their agreement rate would be reported next to the corpus scores.
- **Eval set:** questions collected from real users, or written by Egyptian lawyers who had never seen the corpus. At least two lawyers would label the governing articles, graded as decisive or supporting, with their agreement reported. About 300 questions would be split into a tuning set and a held-out test set, so tuning retrieval can't overfit the score that gets reported.
- **In both cases:** the owner spot-checks a sample against the PDF before any score is used to make a decision.
