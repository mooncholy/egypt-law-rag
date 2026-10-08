# Retrieval misses: hybrid, tuning half; recall@5 0.569 against the 0.90 target

## Missed at 5: 28 of 65

### q001 (rule, en, english)

```text
From what age can a person in Egypt sign contracts and manage their own money without a guardian?
```

- Expected: 44 (not returned)
- Match: all
- Returned: 112, 11, 53, 774, 764, 45, 739, 108, 720, 46
- Legal basis: Art. 44: majority, and with it full capacity, is 21 completed Gregorian years.

### q003 (rule, ar, colloquial)

```text
لو حد خبطني بعربيته وأنا عارف هو مين، قدامي قد إيه أرفع عليه قضية تعويض؟
```

- Expected: 172 (not returned)
- Match: all
- Returned: 50, 983, 444, 207, 483, 362, 10, 51, 317, 221
- Legal basis: Art. 172: a tort claim lapses three years from knowing the injury and the person responsible, fifteen at most.

### q008 (rule, en, english)

```text
Can my neighbour open a window looking straight onto my land from 60 centimetres away?
```

- Expected: 819 (not returned)
- Match: all
- Returned: 812, 809, 924, 928, 803, 618, 1005, 610, 920, 925
- Legal basis: Art. 819: no direct view is allowed at less than one meter.

### q012 (rule, en, english)

```text
If someone pays money they did not actually owe, can they get it back?
```

- Expected: 181 (rank 8)
- Match: all
- Returned: 362, 333, 182, 378, 1144, 185, 183, 181, 727, 187
- Legal basis: Art. 181: what is received without being owed must be returned, unless the payer knew he owed nothing.

### q020 (rule, ar, colloquial)

```text
أنا ساكن في الدور الأرضي واللي ساكن فوقي عايز يعلّي العمارة، من حقه؟
```

- Expected: 861 (rank 6)
- Match: all
- Returned: 1020, 662, 807, 657, 237, 861, 4, 859, 235, 860
- Legal basis: Art. 861: the upper-storey owner may not raise the building in a way that harms the lower storey.

### q024 (rule, en, english)

```text
Can a mortgage contract let the lender take the property automatically if the borrower defaults?
```

- Expected: 1052 (rank 10)
- Match: all
- Returned: 539, 638, 541, 636, 538, 1060, 644, 603, 1072, 1052
- Legal basis: Art. 1052: a clause letting the creditor appropriate the property or sell it without legal formalities is void.

### q030 (rule, en, english)

```text
What can a buyer claim from the seller if a third party proves it owns the whole property sold?
```

- Expected: 443 (not returned)
- Match: all
- Returned: 977, 439, 930, 419, 1063, 441, 924, 331, 803, 925
- Legal basis: Art. 443: on total eviction the buyer recovers the value, fruits returned, useful expenses, costs and damages.

### q050 (rule_with_exception, en, english)

```text
Is a bet on a football match that I am playing in enforceable?
```

- Expected: 740 (rank 8)
- Match: all
- Returned: 739, 199, 756, 750, 1086, 440, 270, 740, 1143
- Legal basis: Art. 740: bets between people personally taking part in sports are excepted from the nullity of Art. 739.

### q053 (multi_article, ar, msa)

```text
ما الفرق بين مدة التقادم العادية ومدة تقادم الحقوق الدورية المتجددة كالأجرة والفوائد؟
```

- Expected: 374–375 (not returned)
- Match: all
- Returned: 969, 385, 970, 1027, 973, 621, 377, 8, 975, 378
- Legal basis: Art. 374: fifteen years; Art. 375: five years for periodic sums such as rent and interest.

### q054 (multi_article, en, english)

```text
A debtor gave away his property to escape his creditors. What must a creditor prove to undo this, and how long does he have?
```

- Expected: 238 (rank 8); 237, 243 (not returned)
- Match: all
- Returned: 239, 207, 246, 260, 241, 258, 787, 238, 1063, 786
- Legal basis: Arts. 237–238 set the conditions of the actio pauliana; Art. 243 sets three years from knowledge, fifteen at most.

### q059 (multi_article, en, english)

```text
What rights does an owner have over his property, and can the state take it from him?
```

- Expected: 802, 805 (not returned)
- Match: all
- Returned: 988, 874, 812, 991, 818, 803, 924, 871, 872, 922
- Legal basis: Art. 802: use, enjoyment and disposal; Art. 805: no deprivation except as the law provides and against fair compensation.

### q060 (multi_article, ar, colloquial)

```text
أنا ساكن بالإيجار والشقة محتاجة تصليحات، مين المسئول عنها، ولو صاحب البيت ما صلحش أعمل إيه؟
```

- Expected: 567 (rank 6); 568 (rank 9)
- Match: all
- Returned: 597, 177, 582, 925, 387, 567, 377, 614, 568, 768
- Legal basis: Art. 567: the lessor makes necessary repairs; Art. 568: if he delays after notice, the lessee may repair and deduct.

### q062 (multi_article, ar, msa)

```text
ما مسئولية صاحب العمل عن خطأ موظفه أثناء العمل، وهل يستطيع أن يرجع على الموظف بما دفعه؟
```

- Expected: 174–175 (not returned)
- Match: all
- Returned: 691, 688, 657, 649, 650, 676, 167, 696, 690, 685
- Legal basis: Art. 174: the master is liable for the servant's acts in the course of employment; Art. 175: he has recourse against the servant.

### q064 (multi_article, ar, colloquial)

```text
أبويا اتوفى وعليه ديون، الديون تتدفع منين، والورثة ياخدوا إمتى نصيبهم؟
```

- Expected: 893 (rank 7); 899 (not returned)
- Match: all
- Returned: 896, 891, 898, 895, 913, 513, 893, 894, 499, 476
- Legal basis: Art. 893: estate debts are paid from estate assets; Art. 899: heirs take the residue after debts.

### q071 (cross_reference, en, english)

```text
If a lease ends but the tenant stays on and the landlord says nothing, for how long is the lease renewed and how is it ended?
```

- Expected: 599 (rank 1); 563 (not returned)
- Match: all
- Returned: 599, 598, 600, 608, 567, 601, 559, 560, 633, 590
- Legal basis: Art. 599: tacit renewal for an indefinite duration, governed by Art. 563 notice periods.

### q072 (cross_reference, ar, msa)

```text
على أي أساس يقدّر القاضي التعويض عن الفعل الضار؟
```

- Expected: 170 (rank 1); 221–222 (not returned)
- Match: all
- Returned: 170, 171, 224, 214, 169, 172, 219, 223, 216, 217
- Legal basis: Art. 170 refers to Arts. 221 and 222 (the English text prints '221 and 22').

### q087 (repealed, en, english)

```text
What does the Civil Code say about how associations are founded and registered?
```

- Expected: 54–80 (not returned)
- Match: any
- Returned: 868, 11, 779, 864, 701, 506, 52, 507, 947, 20
- Legal basis: Articles 54–80 (associations and foundations) are printed as repealed; the subject is governed by special legislation.

### q089 (repealed, en, english)

```text
What are the Civil Code rules on proving a contract by witness testimony?
```

- Expected: 389–417 (not returned)
- Match: any
- Returned: 137, 125, 108, 545, 507, 658, 129, 682, 104, 916
- Legal basis: Articles 389–417 (proof) are printed as repealed; proof is governed by the Law of Evidence.

### q098 (one_language_only, ar, msa)

```text
من يُعتبر حائزا للعقار المرهون في أحكام الرهن الرسمي؟
```

- Expected: 1060 (not returned)
- Match: all
- Returned: 1033, 1032, 1075, 1035, 1044, 1052, 1030, 1084, 1061, 1051
- Legal basis: Art. 1060: the definition of a third party holder is printed in English only.

### q100 (one_language_only, ar, colloquial)

```text
مين يدفع مصاريف صيانة حق المرور اللي على أرضي لصالح أرض جاري؟
```

- Expected: 1021–1022 (not returned)
- Match: any
- Returned: 812, 1038, 925, 919, 1116, 811, 923, 924, 673, 813
- Legal basis: The rule is in Arabic as Art. 1021(2)–(3) and in English as Art. 1022.

### q101 (lay_term, en, english)

```text
Is my employer responsible if a colleague injures a customer while doing his job?
```

- Expected: 174 (not returned)
- Match: all
- Returned: 692, 164, 711, 688, 173, 676, 192, 685, 167, 661
- Legal basis: Art. 174: 'master and servant' is the Code's term for employer and employee.

### q104 (lay_term, ar, colloquial)

```text
سبت شنطة أمانة عند صاحبي وضاعت منه، هو مسئول؟
```

- Expected: 720 (not returned)
- Match: all
- Returned: 176, 164, 1081, 623, 177, 447, 1103, 990, 665, 583
- Legal basis: Art. 720: a gratuitous depositary owes the care he gives his own affairs.

### q110 (lay_term, ar, colloquial)

```text
ورثت أنا وإخواتي بيت ومحدش عايز يقسم غيري، أقدر أجبرهم؟
```

- Expected: 834 (not returned)
- Match: all
- Returned: 849, 322, 903, 853, 314, 902, 845, 913, 905, 908
- Legal basis: Art. 834: every co-owner may demand partition unless bound by law or by an agreement of at most five years.

### q132 (repealed, ar, msa)

```text
ما قواعد القانون المدني في إثبات العقد بشهادة الشهود؟
```

- Expected: 389–417 (not returned)
- Match: any
- Returned: 101, 89, 108, 137, 109, 95, 433, 19, 90, 105
- Legal basis: Articles 389–417 (proof) are printed as repealed; proof is governed by the Law of Evidence.

### q136 (rule_with_exception, en, english)

```text
Can a landlord end a lease because he now needs the property for himself?
```

- Expected: 607 (not returned)
- Match: all
- Returned: 605, 601, 604, 608, 569, 565, 599, 600, 584, 603
- Legal basis: Art. 607: only if so agreed, and with notice under Art. 563.

### q138 (multi_article, en, english)

```text
How far is an employer liable for an employee's fault at work, and can he recover what he paid from the employee?
```

- Expected: 174–175 (not returned)
- Match: all
- Returned: 692, 676, 657, 665, 649, 663, 696, 672, 686, 682
- Legal basis: Art. 174: the master is liable for the servant's acts in the course of employment; Art. 175: he has recourse against the servant.

### q139 (multi_article, en, english)

```text
My father died owing debts. What are the debts paid from, and when do the heirs receive their shares?
```

- Expected: 893 (rank 4); 899 (not returned)
- Match: all
- Returned: 895, 513, 913, 893, 891, 898, 896, 894, 897, 378
- Legal basis: Art. 893: estate debts are paid from estate assets; Art. 899: heirs take the residue after debts.

### q140 (cross_reference, en, english)

```text
On what basis does a judge assess compensation for a harmful act?
```

- Expected: 222 (rank 1); 170, 221 (not returned)
- Match: all
- Returned: 222, 220, 171, 172, 696, 216, 495, 229, 219, 180
- Legal basis: Art. 170 refers to Arts. 221 and 222 (the English text prints '221 and 22').

## Pairs that disagree on the top article: 8 of 11

### p01 (rule)

- q001 (en): top 112, expected 44

```text
From what age can a person in Egypt sign contracts and manage their own money without a guardian?
```

- q123 (ar): top 44, expected 44

```text
من أي سن يستطيع الشخص في مصر أن يبرم العقود ويدير أمواله بنفسه دون ولي أو وصي؟
```

### p10 (repealed)

- q089 (en): top 137, expected 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417

```text
What are the Civil Code rules on proving a contract by witness testimony?
```

- q132 (ar): top 101, expected 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417

```text
ما قواعد القانون المدني في إثبات العقد بشهادة الشهود؟
```

### p12 (rule)

- q003 (ar): top 50, expected 172

```text
لو حد خبطني بعربيته وأنا عارف هو مين، قدامي قد إيه أرفع عليه قضية تعويض؟
```

- q134 (en): top 378, expected 172

```text
If someone hit me with his car and I know who he is, how long do I have to sue him for compensation?
```

### p14 (rule_with_exception)

- q047 (ar): top 580, expected 607

```text
هل يجوز للمؤجر أن ينهي عقد الإيجار لأنه أصبح في حاجة إلى العين لنفسه؟
```

- q136 (en): top 605, expected 607

```text
Can a landlord end a lease because he now needs the property for himself?
```

### p16 (multi_article)

- q062 (ar): top 691, expected 174, 175

```text
ما مسئولية صاحب العمل عن خطأ موظفه أثناء العمل، وهل يستطيع أن يرجع على الموظف بما دفعه؟
```

- q138 (en): top 692, expected 174, 175

```text
How far is an employer liable for an employee's fault at work, and can he recover what he paid from the employee?
```

### p17 (multi_article)

- q064 (ar): top 896, expected 893, 899

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

- q140 (en): top 222, expected 170, 221, 222

```text
On what basis does a judge assess compensation for a harmful act?
```

### p20 (one_language_only)

- q098 (ar): top 1033, expected 1060

```text
من يُعتبر حائزا للعقار المرهون في أحكام الرهن الرسمي؟
```

- q142 (en): top 1060, expected 1060

```text
Under the rules on mortgages, who counts as the holder of the mortgaged property?
```

## Out of scope: what came back first (5)

- q112: Article 728 (dense 0.453, bm25 2.921, rrf 0.032)

```text
What is the prison sentence for theft in Egypt?
```

- q115: Article 1000 (dense 0.481, bm25 4.041, rrf 0.031)

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

- q120: Article 739 (dense 0.501, bm25 2.543, rrf 0.030)

```text
What is the penalty for writing a cheque that bounces?
```
