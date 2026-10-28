# RSNA Intracranial Aneurysm Detection: public vs private leaderboard numbers

Source: Kaggle SDK `get_leaderboard` (private; `override_public=False`, fetched 2026-10-17) and `kaggle competitions leaderboard --download` (public CSV, same day). Joined on TeamId; 1148 real teams on both sides (plus the host's `perfect_submission.parquet` row, dropped). Merged table: `lb_public_vs_private.csv` (columns: team_id, team, pub_rank, priv_rank, pub_score, priv_score). Scripts: `_shake_script1.py`, `_shake_script2.py` (they expect the CSVs in the scratch dir; kept only as a record of the method). Ranks are list positions (ties are in list order, so rank inside a tie block is whatever Kaggle's order says).

NOTE: the leaderboard was "cleaned" after the deadline (recap post 612213: "disqualified some teams that have violated the rules"). Both CSVs were taken after that, so they show the post-cleaning picture. The recap and the competition page both say 1,147 teams; the CSVs have 1148 rows, so the number of removed teams cannot be recovered from the data.

## Whole-field shake-up
- Spearman rank correlation public vs private: 0.945. Median absolute rank move 37, mean 56.
- Top-N overlap (same team in the top N on both boards): top 10: 8; top 50: 45; top 100: 81.
- Gold: private top 12 (10 + 0.2% of 1147). Every one of them was in the public top 15; 10 of the 12 were in the public top 12.
- Silver (top 5%, 57 teams): all 57 were in the public top 101; 52 of them in the public top 57.
- Bronze (top 10%, 114 teams): 94 of them in the public top 114; 12 came from beyond public rank 150 and 4 from beyond 300 (best: public 710 -> private 69).
- Public top 10 -> private ranks: 2, 5, 1, 11, 4, 3, 7, 10, 6, 14. Public top 20 never fell below private 25.
- Private medal lines (score): gold line rank 12 = 0.81888; rank 13 = 0.81766; silver line rank 57 = 0.70729; bronze line rank 114 = 0.69421. Scores fall off a cliff after rank ~25: rank 14 0.8157, 20 0.7912, 25 0.7767, 30 0.751, 40 0.731, 50 0.713, 57 0.707.

## Score gap (private minus public), by public score level
| public score | teams | mean delta | median delta | share ending below the 0.691 copy |
|---|---|---|---|---|
| 0.85-1.0 | 6 | -0.0296 | -0.0285 | 0 |
| 0.80-0.85 | 15 | -0.0186 | -0.0171 | 0 |
| 0.75-0.80 | 13 | -0.0142 | -0.0131 | 0 |
| 0.71-0.75 | 40 | -0.0082 | -0.0080 | 0 |
| 0.70-0.71 | 35 | -0.0106 | -0.0109 | 20% |
| 0.695-0.70 | 50 | -0.0048 | -0.0037 | 20% |
| 0.692-0.695 | 51 | -0.0034 | -0.0022 | 43% |
Higher public score = bigger average loss on the private set (selection effect plus a genuine domain shift, see the 2nd/11th place comments), but the ORDER at the top barely changed.

## The public-notebook tie block
- The most-voted public notebook was "RSNA2025 32ch img infer [LB 0.69]" (730 votes) and the blend/variant notebooks built on it.
- 350 teams had a public score in 0.6905-0.6915; 246 of them exactly 0.69126 (public ranks 317-562). Another 124 teams sat at 0.5 (failed or constant submissions).
- Of the 246 exact copies: 229 ended at private 0.69103 (delta -0.0002), 3 at 0.69034, and 14 at other values (0.5966 to 0.6999: they had changed something, up or down).
- The private tie block at 0.69103 held 263 teams and spans private ranks 245-507. So an unmodified copy of the dominant public notebook finished between the 21st and 44th percentile, not in any medal zone (bronze ended at rank 114). Within the block the order follows submission time only loosely (rank correlation 0.5 against the team's last-submission date).
- Only 3 of the 246 exact copies reached the private top 114 (private ranks 78, 88, 95: scores 0.6963-0.6999), i.e. they tweaked something.
- Teams that squeezed small public gains out of the same family (public 0.70-0.71, public ranks ~80-150): median delta -0.011 versus -0.0002 for the pure copy; biggest public-top-100 drops: 100 -> 631, 79 -> 594, 92 -> 603, 95 -> 515. Two of the names in that list are authors of public blend / transfer-learning notebooks of the same family (a public blend-weight notebook "LB.69+"; a "Transfer learning EfficientNet" notebook). Pure LB-probing of a public family bought a few thousandths publicly and cost 0.01-0.03 privately.
- Largest rises: public 710 -> private 69 (0.6749 -> 0.7021), 649 -> 153, 633 -> 143, 535 -> 78, 507 -> 88: teams with a moderate-but-honest model that did not overfit the public split.

## Final top 12 (private) with public rank
1 tomoon33 (pub 3; 0.88285 -> 0.87912), 2 BraveCoWCoW (pub 1; 0.90035 -> 0.86732), 3 BTYND (6; 0.86715 -> 0.85065), 4 Harshit Sheoran (5; 0.87155 -> 0.84750), 5 more CV challenge pls (2; 0.89306 -> 0.84746), 6 Ian, Theo & Bartley (9; 0.84545 -> 0.84007), 7 MIC-DKFZ (7; 0.84675 -> 0.83173), 8 Konni (14; 0.83721 -> 0.82471), 9 Vibes and Genius Trade-Off (12; 0.83884 -> 0.82300), 10 Dieter (8; 0.84575 -> 0.82206), 11 the "<->AB" team (4; 0.87689 -> 0.82201), 12 Vadasz (15; 0.83684 -> 0.81888).
The team names match the write-ups (1st tomoon33, 2nd BraveCoWCoW, 4th Harshit Sheoran, 5th "more CV challenge pls" = HoangHuyen, 6th Ian/Theo/Bartley, 7th MIC-DKFZ, 8th Konni, 9th Vibes and Genius Trade-Off, 11th <->AB = RihanPiggy), so no visible post-hoc reshuffle of the top 11. No 10th-place write-up was found (team "Dieter").

## Biggest movers in the top 100 (public rank -> private rank)
- Gains: 96 -> 59, 89 -> 63, 97 -> 72, 85 -> 62, 66 -> 49, 54 -> 37.
- Drops: 156 -> 734, 110 -> 658, 105 -> 651, 100 -> 631, 79 -> 594, 92 -> 603, 104 -> 615, 115 -> 608, 49 -> 141, 45 -> 71, 32 -> 54, 60 -> 129.
