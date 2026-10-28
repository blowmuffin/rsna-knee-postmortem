# RSNA 2024 Lumbar Spine Degenerative Classification: brief facts and shake-up (the competition before the aneurysm one)

- Slug rsna-2024-lumbar-spine-degenerative-classification (id 71549); $50k; deadline 2024-10-08; 1,874 teams, 2,423 participants, 23,203 submissions (recap 539766). Notebook-only; metric RSNA lumbar weighted log loss (LOWER is better; private winner 0.3889).
- The recap post carries the same sentence as the aneurysm one ("We've cleaned the leaderboard and disqualified some teams that have violated the rules"). It is Kaggle boilerplate: no list, no reasons, no external-data finding.
- Write-ups read: 1st (NANACHI, 540091), 2nd (Ian Pan / Yuji / Bartley / Kevin, 539452), 3rd (539453), 4th (tattaka / yu4u, 539443). Files l_1st.md .. l_4th.md.
- All top-4 are two-stage: localise (slice index + keypoint coordinates, trained on the host's coordinates plus a community "coordinate pretraining dataset") -> crops around the point -> small 2D/2.5D CNN or MIL / transformer-attention over slices, ensembles of small models. 1st: attention-MIL + bi-LSTM + auxiliary losses (MIL gain on LB 0.0141 public, 0.0197 private, larger than the CV gain of <0.012). 2nd: removed training samples whose label disagreed with the team's OOF ensemble by >=0.8 (+1% public and private) and multi-crop test-time aggregation ("27 crops per target"). Simple blends and tiny post-processing (x1.25 on the max spinal-severe) in 2nd.

## Leaderboard shake-up (same method as the aneurysm one; 1875 teams in both files)
- Spearman public vs private rank: 0.975; median absolute rank move 17.
- Top-10 overlap 8, top-50 overlap 39, top-100 overlap 84. Private top 10 all came from public top 13; private top 12 from public top 15.
- Private winner was public rank 5 (0.3392 -> 0.3889). Mean private-minus-public log loss for the public top 30: +0.060 (everyone is worse on private, the order is mostly kept).
- Largest public tie block: 437 teams at public 0.481 (a public notebook), public ranks 193-629. Their private scores clustered at 0.5323 (275 of them identical); private rank median 417, 5th-95th percentile 212-633; 11 reached the bronze zone, none the silver zone (private top 94). A 278-team private block at 0.532335 spans ranks 301-578; its rank vs last-submission-date correlation is 0.54.
- Biggest public-top-100 drops: 96 -> 719, 61 -> 246, 99 -> 184, 73 -> 141, 46 -> 98, 38 -> 95. Gains: 82 -> 56, 79 -> 54, 69 -> 47.
- Medal lines by Kaggle's formula: gold 13, silver 93, bronze 187.
