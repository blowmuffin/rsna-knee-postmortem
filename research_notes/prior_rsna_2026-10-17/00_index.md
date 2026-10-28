# Prior RSNA competition research, 2026-10-17 (agent 3 of 3)

Question: what happened in the RSNA Kaggle competition that ran just before RSNA Knee, and what transfers to our last 5 days.

## Identification (evidence)
Search of all RSNA-hosted competitions by deadline (Kaggle MCP `search_competitions`, query "RSNA"): after the knee one (deadline 2026-10-22) the latest is RSNA Intracranial Aneurysm Detection, slug `rsna-intracranial-aneurysm-detection`, id 99552, 2025-07-28 to 2025-10-14, 1,147 teams, $50,000, notebook-only. Before it: RSNA 2024 Lumbar Spine Degenerative Classification (id 71549, deadline 2024-10-08, 1,874 teams). Both share hosts (RSNA), notebook-only format and the same Kaggle staff (Ryan Holbrook).

## Files
Competition pages (saved verbatim from the API): `_page_rules.md`, `_page_Description.md`, `_page_Evaluation.md`, `_page_Timeline.md`, `_page_data-description.md`, `_page_Prizes.md`, `_page_Code Requirements.md`, `_page_abstract.md`, `_page_Acknowledgements.md`.

Write-ups (author, topic id): `w01_1st.md` tomoon33 (611846), `w02_2nd.md` BraveCoWCoW / Pengcheng Shi (611867), `w03_3rd.md` TmT et al. (611856), `w04_4th.md` Harshit Sheoran (611893), `w05_5th.md` HoangHuyen (611849), `w06_6th.md` Theo Viel / Ian Pan / Bartley (611925), `w07_7th.md` MIC-DKFZ (612039), `w08_8th.md` Konni (613534), `w09_9th.md` Vibes and Genius Trade-Off (611908), `w11_11th.md` RihanPiggy (612186), `w15_15th.md` (612775), `w18_18th.md` (612006), `w38_38th.md` (611907), `w55_55th.md` (611858), `w57_57th.md` (685172). There is NO 10th-place write-up (the 10th team, "Dieter", was outside the 9 paid places, and none was found in the 201-topic forum listing).

Discussions: `d_recap.md` (612213 final recap), `d_wrap_delays.md` (611953 host statement on submission delays, no extension), `d_data_issues.md` (610101 no more data fixes), `d_slower.md` (611482 runtime variance and timeouts), `d_server_load.md` (611721), `d_inference_time_q.md` (607923 staff answers on the 12 h limit), `d_rescoring.md` (600908), `d_rerun_done.md` (603990), `d_external_list.md` (597777), `d_dataset_bad.md` (611939), `d_cow_seg.md` (612246). Read via MCP and summarised in `d_mcp_read_topics.md`: 611750 (final selection), 608179 (external data ruling), 611652 (shake-up question), 603679 (CV vs LB), 602885 (scoring update).

Leaderboards: `lb_public_vs_private.csv` (both boards joined, aneurysm), `shakeup_numbers.md` (all the numbers), `_shake_script1.py`, `_shake_script2.py` (method). Lumbar: `lumbar_2024_shakeup.md`, and write-ups `l_1st.md`, `l_2nd.md`, `l_3rd.md`, `l_4th.md`, `l_recap.md`.

Deliverable built from these: `reports/2026-10-17 prior RSNA competition.md`.

## Method notes
- Kaggle SDK `ListTopics` returned 403 here; topic lists came from the MCP `list_competition_topics` (201 topics, all pages of "Top" 1-4 and "Recent" 1-2 seen). `get_topic`/`list_comments` worked but hit 429 whenever other agents were fetching; a tolerant retry loop was used.
- Private leaderboard from SDK `get_leaderboard` (override_public false); public from `kaggle competitions leaderboard --download`. No competition data was downloaded; nothing was posted, voted, joined or submitted.
