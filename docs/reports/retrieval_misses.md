# Retrieval misses: hybrid, tuning half; recall@5 0.708 against the 0.90 target

## Missed at 5: 19 of 65

### q001 (rule, en, english)

```text
From what age can a person in Egypt sign contracts and manage their own money without a guardian?
```

- Expected: 44 (rank 8)
- Match: all
- Returned: 112, 11, 53, 734, 173, 45, 46, 44, 108, 727
- Legal basis: Art. 44: majority, and with it full capacity, is 21 completed Gregorian years.

### q003 (rule, ar, colloquial)

```text
لو حد خبطني بعربيته وأنا عارف هو مين، قدامي قد إيه أرفع عليه قضية تعويض؟
```

- Expected: 172 (not returned)
- Match: all
- Returned: 51, 575, 171, 170, 50, 221, 164, 222, 175, 216
- Legal basis: Art. 172: a tort claim lapses three years from knowing the injury and the person responsible, fifteen at most.

### q053 (multi_article, ar, msa)

```text
ما الفرق بين مدة التقادم العادية ومدة تقادم الحقوق الدورية المتجددة كالأجرة والفوائد؟
```

- Expected: 375 (rank 1); 374 (not returned)
- Match: all
- Returned: 375, 385, 969, 379, 378, 377, 386, 380, 8, 382
- Legal basis: Art. 374: fifteen years; Art. 375: five years for periodic sums such as rent and interest.

### q054 (multi_article, en, english)

```text
A debtor gave away his property to escape his creditors. What must a creditor prove to undo this, and how long does he have?
```

- Expected: 238 (rank 1); 237, 243 (not returned)
- Match: all
- Returned: 238, 258, 235, 260, 786, 1063, 239, 246, 340, 273
- Legal basis: Arts. 237–238 set the conditions of the actio pauliana; Art. 243 sets three years from knowledge, fifteen at most.

### q057 (multi_article, ar, msa)

```text
كيف تتم قسمة المال الشائع إذا لم يتفق الشركاء، ومتى يُباع بالمزاد؟
```

- Expected: 841 (rank 2); 836 (rank 6)
- Match: all
- Returned: 842, 841, 849, 905, 846, 836, 834, 835, 827, 847
- Legal basis: Art. 836: partition through the Summary Court with experts; Art. 841: sale if division in kind is impossible or harmful.

### q062 (multi_article, ar, msa)

```text
ما مسئولية صاحب العمل عن خطأ موظفه أثناء العمل، وهل يستطيع أن يرجع على الموظف بما دفعه؟
```

- Expected: 174–175 (not returned)
- Match: all
- Returned: 192, 167, 692, 163, 165, 696, 690, 217, 665, 658
- Legal basis: Art. 174: the master is liable for the servant's acts in the course of employment; Art. 175: he has recourse against the servant.

### q064 (multi_article, ar, colloquial)

```text
أبويا اتوفى وعليه ديون، الديون تتدفع منين، والورثة ياخدوا إمتى نصيبهم؟
```

- Expected: 893 (rank 1); 899 (rank 9)
- Match: all
- Returned: 893, 895, 898, 896, 891, 897, 884, 894, 899, 913
- Legal basis: Art. 893: estate debts are paid from estate assets; Art. 899: heirs take the residue after debts.

### q072 (cross_reference, ar, msa)

```text
على أي أساس يقدّر القاضي التعويض عن الفعل الضار؟
```

- Expected: 170 (rank 1); 221 (rank 3); 222 (not returned)
- Match: all
- Returned: 170, 171, 221, 169, 216, 168, 214, 224, 173, 192
- Legal basis: Art. 170 refers to Arts. 221 and 222 (the English text prints '221 and 22').

### q087 (repealed, en, english)

```text
What does the Civil Code say about how associations are founded and registered?
```

- Expected: 54–80 (not returned)
- Match: any
- Returned: 52, 11, 506, 44, 505, 864, 863, 30, 701, 779
- Legal basis: Articles 54–80 (associations and foundations) are printed as repealed; the subject is governed by special legislation.

### q089 (repealed, en, english)

```text
What are the Civil Code rules on proving a contract by witness testimony?
```

- Expected: 389–417 (not returned)
- Match: any
- Returned: 137, 104, 125, 126, 97, 102, 129, 117, 127, 698
- Legal basis: Articles 389–417 (proof) are printed as repealed; proof is governed by the Law of Evidence.

### q098 (one_language_only, ar, msa)

```text
من يُعتبر حائزا للعقار المرهون في أحكام الرهن الرسمي؟
```

- Expected: 1060 (not returned)
- Match: all
- Returned: 1075, 1084, 1063, 1032, 1071, 1033, 1051, 1036, 1061, 1056
- Legal basis: Art. 1060: the definition of a third party holder is printed in English only.

### q100 (one_language_only, ar, colloquial)

```text
مين يدفع مصاريف صيانة حق المرور اللي على أرضي لصالح أرض جاري؟
```

- Expected: 1022 (rank 6); 1021 (not returned)
- Match: any
- Returned: 812, 614, 567, 1116, 808, 1022, 809, 936, 623, 640
- Legal basis: The rule is in Arabic as Art. 1021(2)–(3) and in English as Art. 1022.

### q101 (lay_term, en, english)

```text
Is my employer responsible if a colleague injures a customer while doing his job?
```

- Expected: 174 (rank 8)
- Match: all
- Returned: 167, 166, 173, 175, 164, 168, 711, 174, 192, 696
- Legal basis: Art. 174: 'master and servant' is the Code's term for employer and employee.

### q104 (lay_term, ar, colloquial)

```text
سبت شنطة أمانة عند صاحبي وضاعت منه، هو مسئول؟
```

- Expected: 720 (not returned)
- Match: all
- Returned: 983, 984, 207, 990, 1103, 727, 504, 164, 269, 166
- Legal basis: Art. 720: a gratuitous depositary owes the care he gives his own affairs.

### q110 (lay_term, ar, colloquial)

```text
ورثت أنا وإخواتي بيت ومحدش عايز يقسم غيري، أقدر أجبرهم؟
```

- Expected: 834 (rank 8)
- Match: all
- Returned: 903, 845, 913, 907, 853, 908, 910, 834, 905, 902
- Legal basis: Art. 834: every co-owner may demand partition unless bound by law or by an agreement of at most five years.

### q132 (repealed, ar, msa)

```text
ما قواعد القانون المدني في إثبات العقد بشهادة الشهود؟
```

- Expected: 389–417 (not returned)
- Match: any
- Returned: 137, 773, 125, 104, 90, 95, 147, 102, 127, 145
- Legal basis: Articles 389–417 (proof) are printed as repealed; proof is governed by the Law of Evidence.

### q138 (multi_article, en, english)

```text
How far is an employer liable for an employee's fault at work, and can he recover what he paid from the employee?
```

- Expected: 175 (rank 2); 174 (not returned)
- Match: all
- Returned: 165, 175, 672, 696, 665, 192, 977, 171, 695, 692
- Legal basis: Art. 174: the master is liable for the servant's acts in the course of employment; Art. 175: he has recourse against the servant.

### q139 (multi_article, en, english)

```text
My father died owing debts. What are the debts paid from, and when do the heirs receive their shares?
```

- Expected: 893 (rank 5); 899 (rank 7)
- Match: all
- Returned: 895, 896, 378, 913, 893, 897, 899, 891, 898, 894
- Legal basis: Art. 893: estate debts are paid from estate assets; Art. 899: heirs take the residue after debts.

### q140 (cross_reference, en, english)

```text
On what basis does a judge assess compensation for a harmful act?
```

- Expected: 170 (rank 1); 222 (rank 5); 221 (rank 7)
- Match: all
- Returned: 170, 171, 168, 164, 222, 216, 221, 214, 224, 229
- Legal basis: Art. 170 refers to Arts. 221 and 222 (the English text prints '221 and 22').

## Pairs that disagree on the top article: 4 of 11

### p12 (rule)

- q003 (ar): top 51, expected 172

```text
لو حد خبطني بعربيته وأنا عارف هو مين، قدامي قد إيه أرفع عليه قضية تعويض؟
```

- q134 (en): top 172, expected 172

```text
If someone hit me with his car and I know who he is, how long do I have to sue him for compensation?
```

### p16 (multi_article)

- q062 (ar): top 192, expected 174, 175

```text
ما مسئولية صاحب العمل عن خطأ موظفه أثناء العمل، وهل يستطيع أن يرجع على الموظف بما دفعه؟
```

- q138 (en): top 165, expected 174, 175

```text
How far is an employer liable for an employee's fault at work, and can he recover what he paid from the employee?
```

### p17 (multi_article)

- q064 (ar): top 893, expected 893, 899

```text
أبويا اتوفى وعليه ديون، الديون تتدفع منين، والورثة ياخدوا إمتى نصيبهم؟
```

- q139 (en): top 895, expected 893, 899

```text
My father died owing debts. What are the debts paid from, and when do the heirs receive their shares?
```

### p20 (one_language_only)

- q098 (ar): top 1075, expected 1060

```text
من يُعتبر حائزا للعقار المرهون في أحكام الرهن الرسمي؟
```

- q142 (en): top 1032, expected 1060

```text
Under the rules on mortgages, who counts as the holder of the mortgaged property?
```

## Out of scope: what came back first (5)

- q112: Article 11 (dense 0.431, bm25 2.907, rrf 0.030)

```text
What is the prison sentence for theft in Egypt?
```

- q115: Article 701 (dense 0.493, bm25 none, rrf 0.016)

```text
إزاي أعمل توكيل في الشهر العقاري وبكام؟
```

- q116: Article 252 (dense 0.565, bm25 4.506, rrf 0.032)

```text
What is the deadline to appeal a civil court judgment?
```

- q119: Article 599 (dense 0.535, bm25 3.701, rrf 0.033)

```text
هو الإيجار القديم هيتلغي إمتى؟
```

- q120: Article 260 (dense 0.435, bm25 none, rrf 0.014)

```text
What is the penalty for writing a cheque that bounces?
```
