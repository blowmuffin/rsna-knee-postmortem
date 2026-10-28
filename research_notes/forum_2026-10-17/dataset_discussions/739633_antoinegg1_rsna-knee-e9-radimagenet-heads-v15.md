# Provenance and reuse terms for the v15 training recipe
739633 | dataset antoinegg1/rsna-knee-e9-radimagenet-heads-v15 | Michael O'Flaherty | 2026-09-05T02:51 | votes 0

Thank you for publishing the v15 trainer and contracts. Could you share a small source/provenance receipt for the run that produced heads SHA-256
0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa?

The remaining bindings we need are the producing notebook/script version and its upstream helper source, effective SLICE_BAND and other cache globals, seed/runtime versions, and the dataset versions plus file SHA-256 values for report_labels_v2.csv, llm_labels_v2.csv and labels_llm_gpt56sol.csv. Aggregate hashes/counts for study membership and report-group folds would also help. We do not need label rows, reports, images, weights or OOF predictions.

Could you also confirm the applicable terms for adapting the published trainer in private competition research? Our intended validation would exclude each outer fold from fitting and use training-only checkpoint selection.

## Comments (none as of 10-17 ~16:30 UTC)