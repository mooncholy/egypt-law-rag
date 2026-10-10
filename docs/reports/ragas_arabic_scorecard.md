# RAGAS on Arabic answers: what to trust

Which of the four RAGAS scores can be trusted on this project's Arabic answers, with this judge, and for what use. The handbook asks for this note before RAGAS scores drive decisions (Module 5, Step 14).

**Setup.** Generator `qwen2.5:3b-instruct-fp16` (Ollama), judge `gpt-oss:120b-cloud` (D23), embeddings `BAAI/bge-m3`, prompt v2, the `champion` retrieval config. The answer set is the 65 in-scope questions of the tuning half: 31 Arabic (9 of them colloquial) and 34 English (`docs/reports/evaluation_data.md`).

**Evidence.** Two kinds:

- **Judge noise, measured.** `champion` was scored twice (C20). All 65 answers were word for word identical, so any difference between the two runs' scores is the judge's alone.
- **A reading.** Every answer `champion` gave was read against its governing articles and marked faithful or not, then compared with RAGAS's faithfulness score (0.5 as the cut). Refusals and partly faithful answers were left out of the comparison.

## Verdicts

| Score | Arabic | English | Use it for |
| --- | --- | --- | --- |
| Faithfulness | Average only. Per question, the judge moves 0.21 on average between identical runs, and 9 of 23 answers move by more than 0.2. The reading agrees on 15 of 17 answers. | Average only. Per question 0.13; 5 of 28 move by more than 0.2. The reading agrees on 17 of 19. | Comparing configs over the whole set, per language. A single answer's score only points to an answer worth reading. |
| Answer relevancy | Stable (0.03 per question). | Noisier (0.11; 4 of 28 move by more than 0.2). | A secondary signal. |
| Context precision | Stable (0.03). | Stable (0.02). | Choosing retrieval configs. Unscored on 7 of 51 answers (14%), in both languages, when the judge's reply couldn't be parsed. |
| Context recall | Nearly stable (0.06; 4 of 23 move by more than 0.2). | Stable (0.00). | Choosing retrieval configs; the clearest separation between configs (`docs/reports/answer_comparison.md`). |

Over a whole run the noise largely cancels: overall faithfulness differed by 0.0096 between the two runs, within C20's 0.02.

## What the scores miss

1. **Refusals score as unfaithful.** When the generator says the articles don't address the question, faithfulness is usually 0, and sometimes an arbitrary 0.5 (q001, q057). Such an answer isn't unfaithful; it's unhelpful. In `champion`, 11 of the 51 answers that reached the generator were refusals or a bare article label, most with the governing article in front of it (q005, q030, q038, q096, q107, q134, q135). `answer_comparison.md` counts them separately.
2. **Faithful isn't the same as correct.** Faithfulness asks whether each claim follows from the articles given, not whether those articles govern the question. q123 asks the age of majority. Retrieval ranks Article 112 first (a minor of eighteen allowed to manage property), and the answer says eighteen, citing it. That answer is partly faithful (0.33) and legally wrong: Article 44 says twenty-one. q062 answers an employer's liability from Article 192 instead of 174 and scores 0.79. Context recall and the reference answer catch these cases; faithfulness can't.
3. **Arabic verbatim quotes are under-scored.** q131 quotes Article 147 word for word and scores 0.67; the English verbatim quotes (q079, q082, q084) score 1.0.
4. **The generator mixes scripts in Arabic.** It writes `Article 374 | المادة ٣٧٤`, `وفقًا لArticle 607`, and once a corrupted `السcaf`. The judge has to split such answers into statements, which is where most of the Arabic noise is likely to come from.
5. **Arabic faithfulness is lower than English for every config** (0.49 to 0.61 against 0.82 to 0.92 on the questions all configs answered). Generator weakness in Arabic and judge noise both contribute, and these scores can't separate them; a fluent human review of Arabic answers can.

## Recommendations

- **Report faithfulness as an average over at least 50 questions, per language**, and treat a difference under 0.02 between configs as judge noise. Use per-question scores to find answers to read, never to decide.
- **Choose retrieval configs on context precision and context recall**, which are stable in both languages.
- **Count generator refusals separately** (`answer_comparison.md`); they are the largest single cause of low faithfulness here, and a prompt or model change should be judged on them directly.
- **The CI gate (Phase 11) scores 20 questions**, where an average carries more noise than over 65. Measure the gate's own repeatability before fixing its threshold; the current overall faithfulness (0.57) is below the handbook's 0.75.

## Truth point

The reading was done by the same model (Claude Opus 5.5) that wrote the reference answers and labelled the eval set, and no second reader checked it. It is one annotator's judgment, on 36 answers. The noise measurements need no annotator.

## The reading

Faithfulness from the first `champion` run. F: every claim follows from the articles given; U: at least one claim doesn't.

| Question | Language | RAGAS | Reading | Agrees |
| --- | --- | --- | --- | --- |
| q015 | ar | 1.00 | F | yes |
| q018 | ar | 1.00 | F | yes |
| q020 | ar | 0.00 | U (wrong rule, corrupted text) | yes |
| q022 | ar | 1.00 | F | yes |
| q033 | ar | 0.73 | F (answers from Art. 962 rather than 961) | yes |
| q035 | ar | 1.00 | F | yes |
| q039 | ar | 0.00 | F (states Art. 452(2)'s exception) | no |
| q041 | ar | 0.00 | U (answers from Art. 597) | yes |
| q047 | ar | 0.00 | U ("yes" without the required agreement) | yes |
| q053 | ar | 0.33 | U (both periods five years) | yes |
| q062 | ar | 0.79 | F (to Art. 192; the wrong article) | yes |
| q067 | ar | 0.75 | F | yes |
| q069 | ar | 0.67 | F | yes |
| q072 | ar | 1.00 | F | yes |
| q077 | ar | 0.67 | U (cites Art. 457, not given in that sense) | no |
| q130 | ar | 1.00 | F | yes |
| q131 | ar | 0.67 | F (verbatim quote) | yes |
| q008 | en | 0.60 | F (to Art. 820; oblique view) | yes |
| q012 | en | 0.86 | F | yes |
| q024 | en | 0.71 | F | yes |
| q025 | en | 0.71 | F | yes |
| q028 | en | 0.73 | F | yes |
| q032 | en | 1.00 | F | yes |
| q036 | en | 0.40 | U (opens with "the buyer cannot sue") | yes |
| q042 | en | 1.00 | F | yes |
| q045 | en | 1.00 | U (misstates whose share is claimed) | no |
| q059 | en | 1.00 | F | yes |
| q075 | en | 0.67 | U (names Arts. 1048 and 1049) | no |
| q079 | en | 1.00 | F (verbatim quote) | yes |
| q082 | en | 1.00 | F (verbatim quote) | yes |
| q084 | en | 1.00 | F | yes |
| q092 | en | 0.75 | F | yes |
| q136 | en | 1.00 | F | yes |
| q139 | en | 0.75 | F | yes |
| q140 | en | 0.91 | F | yes |
| q142 | en | 0.75 | F (to Arts. 1032 and 1033) | yes |

Left out: refusals and bare labels (q001, q005, q009, q030, q038, q057, q096, q101, q107, q134, q135), and partly faithful answers (q054, q071, q098, q123).
