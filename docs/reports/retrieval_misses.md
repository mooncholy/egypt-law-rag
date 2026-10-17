# Retrieval misses: hybrid, tuning half; recall@5 0.815 against the 0.90 target

## Missed at 5: 12 of 65

### q001 (rule, en, english)

```text
From what age can a person in Egypt sign contracts and manage their own money without a guardian?
```

- Expected: 44 (rank 6)
- Match: all
- Returned: 112, 111, 11, 108, 53, 44, 46, 173, 45, 188
- Legal basis: Art. 44: majority, and with it full capacity, is 21 completed Gregorian years.

### q003 (rule, ar, colloquial)

```text
لو حد خبطني بعربيته وأنا عارف هو مين، قدامي قد إيه أرفع عليه قضية تعويض؟
```

- Expected: 172 (not returned)
- Match: all
- Returned: 575, 51, 164, 175, 171, 50, 170, 221, 222, 960
- Legal basis: Art. 172: a tort claim lapses three years from knowing the injury and the person responsible, fifteen at most.

### q053 (multi_article, ar, msa)

```text
ما الفرق بين مدة التقادم العادية ومدة تقادم الحقوق الدورية المتجددة كالأجرة والفوائد؟
```

- Expected: 375 (rank 1); 374 (not returned)
- Match: all
- Returned: 375, 385, 379, 376, 378, 969, 380, 377, 386, 1027
- Legal basis: Art. 374: fifteen years; Art. 375: five years for periodic sums such as rent and interest.

### q054 (multi_article, en, english)

```text
A debtor gave away his property to escape his creditors. What must a creditor prove to undo this, and how long does he have?
```

- Expected: 239 (rank 1); 237–238, 243 (not returned)
- Match: all
- Returned: 239, 258, 786, 1063, 1051, 260, 340, 788, 790, 787
- Legal basis: Arts. 237–238 set the conditions of the actio pauliana; Art. 239: the creditor need only prove the amount of the debt; Art. 243 sets three years from knowledge, fifteen at most.

### q059 (multi_article, en, english)

```text
What rights does an owner have over his property, and can the state take it from him?
```

- Expected: 802 (rank 2); 805 (rank 7)
- Match: all
- Returned: 988, 802, 803, 1078, 991, 874, 805, 812, 1141, 804
- Legal basis: Art. 802: use, enjoyment and disposal; Art. 805: no deprivation except as the law provides and against fair compensation.

### q062 (multi_article, ar, msa)

```text
ما مسئولية صاحب العمل عن خطأ موظفه أثناء العمل، وهل يستطيع أن يرجع على الموظف بما دفعه؟
```

- Expected: 175 (rank 2); 174 (not returned)
- Match: all
- Returned: 192, 175, 690, 692, 324, 167, 696, 217, 665, 691
- Legal basis: Art. 174: the master is liable for the servant's acts in the course of employment; Art. 175: he has recourse against the servant.

### q098 (one_language_only, ar, msa)

```text
من يُعتبر حائزا للعقار المرهون في أحكام الرهن الرسمي؟
```

- Expected: 1060 (not returned)
- Match: all
- Returned: 1033, 1084, 1061, 1075, 1118, 1036, 1148, 1109, 1032, 1030
- Legal basis: Art. 1060: the definition of a third party holder is printed in English only.

### q100 (one_language_only, ar, colloquial)

```text
مين يدفع مصاريف صيانة حق المرور اللي على أرضي لصالح أرض جاري؟
```

- Expected: 1022 (rank 7); 1021 (not returned)
- Match: any
- Returned: 812, 614, 989, 1116, 808, 567, 1022, 918, 809, 810
- Legal basis: The rule is in Arabic as Art. 1021(2)–(3) and in English as Art. 1022.

### q104 (lay_term, ar, colloquial)

```text
سبت شنطة أمانة عند صاحبي وضاعت منه، هو مسئول؟
```

- Expected: 720 (not returned)
- Match: all
- Returned: 983, 984, 207, 990, 1103, 166, 164, 1081, 504, 727
- Legal basis: Art. 720: a gratuitous depositary owes the care he gives his own affairs.

### q110 (lay_term, ar, colloquial)

```text
ورثت أنا وإخواتي بيت ومحدش عايز يقسم غيري، أقدر أجبرهم؟
```

- Expected: 834 (rank 6)
- Match: all
- Returned: 903, 853, 913, 895, 907, 834, 908, 902, 845, 905
- Legal basis: Art. 834: every co-owner may demand partition unless bound by law or by an agreement of at most five years.

### q138 (multi_article, en, english)

```text
How far is an employer liable for an employee's fault at work, and can he recover what he paid from the employee?
```

- Expected: 175 (rank 4); 174 (not returned)
- Match: all
- Returned: 977, 696, 192, 175, 665, 692, 768, 221, 690, 695
- Legal basis: Art. 174: the master is liable for the servant's acts in the course of employment; Art. 175: he has recourse against the servant.

### q139 (multi_article, en, english)

```text
My father died owing debts. What are the debts paid from, and when do the heirs receive their shares?
```

- Expected: 899 (rank 2); 893 (rank 8)
- Match: all
- Returned: 895, 899, 913, 378, 302, 896, 894, 893, 898, 897
- Legal basis: Art. 893: estate debts are paid from estate assets; Art. 899: heirs take the residue after debts.

## Pairs that disagree on the top article: 5 of 11

### p12 (rule)

- q003 (ar): top 575, expected 172

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

- q138 (en): top 977, expected 174, 175

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

### p18 (cross_reference)

- q072 (ar): top 170, expected 170, 221, 222

```text
على أي أساس يقدّر القاضي التعويض عن الفعل الضار؟
```

- q140 (en): top 171, expected 170, 221, 222

```text
On what basis does a judge assess compensation for a harmful act?
```

### p20 (one_language_only)

- q098 (ar): top 1033, expected 1060

```text
من يُعتبر حائزا للعقار المرهون في أحكام الرهن الرسمي؟
```

- q142 (en): top 1032, expected 1060

```text
Under the rules on mortgages, who counts as the holder of the mortgaged property?
```

## Out of scope: what came back first (5)

- q112: Article 728 (dense 0.442, bm25 2.622, rrf 0.030)

```text
What is the prison sentence for theft in Egypt?
```

- q115: Article 1000 (dense 0.527, bm25 3.922, rrf 0.032)

```text
إزاي أعمل توكيل في الشهر العقاري وبكام؟
```

- q116: Article 252 (dense 0.619, bm25 5.099, rrf 0.032)

```text
What is the deadline to appeal a civil court judgment?
```

- q119: Article 599 (dense 0.539, bm25 3.844, rrf 0.032)

```text
هو الإيجار القديم هيتلغي إمتى؟
```

- q120: Article 889 (dense 0.435, bm25 none, rrf 0.014)

```text
What is the penalty for writing a cheque that bounces?
```
