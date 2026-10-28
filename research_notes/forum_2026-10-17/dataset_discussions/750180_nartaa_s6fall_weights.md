# Materials needed for leakage-controlled retraining and grouped CV
750180 | 李鲲鹏 (@ahmmmmmm7) | 2026-10-17T05:22 | votes 0

Thank you for the detailed releases. I am trying to evaluate the complete training procedure with grouped outer folds, rather than score the released all-data weights as OOF.

I recovered the public HPO label/fold assets and reproduced labels_v1_blend exactly. I also recovered the six Heliosli base scripts from s6_extras.bin and verified their hashes. Could you publicly release, or identify the existing public location of:

- the S6 training adapter with SHA256 ae18d2e11de9a16202934787a9428ecdcefc0543d7944a75182926fb53965967;

- the construction procedure for targets_s6f0.csv, including the unaddressed-cell prior fill and per-cell weights;

- pool_ok.json (SHA256 378f782f6dd0a613df8bbdbc6d3d9f38d6b70697ef966b178800ea350a83e9a6), or a deterministic procedure and exclusion reasons that reproduce its 4263-study membership;

- confirmation of whether the folds_v0.json in rsna-knee-hpo-assets v9 is the one used for S6;

- the selected 384/224 CoAtNet training source/configuration, including masked OAI loss and SWA selection, and the permitted route for qualified researchers to obtain the exact OAI linkage manifest?

I am not requesting redistribution of restricted OAI images or subject-level data outside their permitted access process. The existing Gold58 scores are treated as development diagnostics, not fresh validation.

## Comments
- [Danial Zakaria | 2026-10-17T06:03 | v0] Correction to point 3. The 86 studies missing from the pool were not removed by the hosts. We deleted their DICOMs locally on 2026-09-29 to free disk space and never downloaded them again. They are still in the competition data. So the pool can't be rebuilt from the competition files: training_code/s6/data/pool_ok.json is the authoritative membership, and the 86 are simply folds_v0 folds 0–4 minus that list. It wasn't a quality filter, and none of the 86 were in Gold-58. The dataset README has been corrected.
  
  Also, under the competition's public code-sharing rule, everything in training_code/ is shared under Apache-2.0 (stated in the README now). The same code is linked from the competition forum.
- [Danial Zakaria | 2026-10-17T06:01 | v0] Thanks for doing this properly. Retraining under grouped outer folds is the right test, and treating Gold-58 as a development diagnostic is right too. I've added everything I can share to this dataset under training_code/. Every file is byte-identical to what ran, and all of them are listed in training_code/SHA256SUMS. Answers in your order:
  
  1. Training adapter ae18d2e1…
  It is training_code/s6/train_s6f0_all.py: Heliosli's compact_train.py, taken from his public notebook (Apache-2.0), with two sets of patches.
  
  - P1–P8 (fold-0 run, train_s6f0.py, 6c0118fa…):
  - on-the-fly DICOM→rows loading instead of his pixel cache, using his own prepare/geometry path and SKM-TEA anchors precomputed once per study (precompute_anchors.py);
  
  - a frozen --pool;
  
  - --expect-train/--expect-holdout split guards;
  
  - atomic per-epoch checkpoints with --resume.
  
  - A1–A4 (--all-data): trains the fold-0 iteration on every pool study minus gold, with nothing held out. It refuses to run unless the pool minus gold equals the training set and the hold-out is 0.
  
  compact_train_to_train_s6f0_all.diff shows every change. Everything above main() is his code unchanged, so Net, the loss and deploy_state_dict are his. The production argv is in run_all_alldata.sh, and inputs.sha256 pins every training input.
  
  The final weights are taken at epoch 16. There is no SWA and no EMA, and nothing was selected on Gold-58 for S6. Heliosli's cloud_code and anatomy.pt are not copied (no licence is stated on his dataset). Attach heliosli/rsna-orthofoundation-fold0-epoch7-20261004 yourself; heliosli.SHA256SUMS and cloud_code.SHA256SUMS pin the exact files used.
  
  2. targets_s6f0.csv (s6/make_data.py, file sha 8006cc39…): there is no prior fill and there are no per-cell weights
  
  - Every non-gold row has all 12 L|prior cells set to labels_v1_blendjev. All L|strict cells are blank. Arm E_prior therefore uses a weight of 1 in every cell with compact_train's weight-normalised BCE. The bias init is the logit of the mean prior over the training set.
  
  - The 58 gold rows are is_gold=1 with blank labels, so the trainer drops them.
  
  - One catch: S6 used labels_v1_blendjev (c97ee6b1…), not labels_v1_blend. It is the cell-wise mean of labels_v1_blend and riadmohamed42's JEV HYBRID_labels_scores.csv (riadmohamed42/jev-knee-labels, CC0); where the JEV cell is NaN, the blend value is kept. training_code/labels/ has the parquet and build_blendjev.py, which rebuilds it from your labels_v1_blend with max |diff| = 0.
  
  3. pool_ok.json (s6/freeze_pool.py): it has 4,321 studies, not 4,263
  4,321 = 4,263 training studies + the 58 gold studies, which are kept for inference. The trainer uses pool − gold. To rebuild it:
  
  - take the folds_v0 studies with fold 0–4 (4,349), all of which have a blendjev row;
  
  - drop the 86 studies the hosts removed in the 2026-09-29 data update (71 from folds 1–4 and 15 from fold 0);
  
  - add the gold 58.
  
  The anchor precompute failed on 0 of 4,321 studies (pool_ok.counts.json), so no study was excluded for quality.
  
  4. folds_v0.json
  Yes, it is the one in rsna-knee-hpo-assets: sha256 12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae, 312,966 bytes. Fold sizes are 870/870/870/870/869, with gold at −1. Note that it holds only study → fold and has no patient/group key, so your grouped outer folds have to come from your own grouping.
  
  5. CoAtNet 384 / 224 and OAI (training_code/coatnet/, details in RUNS.md)
  
  - Both models use train_knee.py (our edit of dreaddevelopment's public Raptor training notebook), run unchanged through ext_trainer.py. The wrapper loads the trainer by path and patches four globals:
  - 2,399 OAI knees are appended to the training split only, giving 6,748 rows;
  
  - external cells supervise only PF OA, Lateral OA, Synovitis, and every other external cell is NaN, i.e. masked out of the loss;
  
  - pos_weight is pinned to the RSNA labels.
  
  - 384 (behind 0.948/0.949): coatnet_rmlp_2_rw_384, crop 1.0. 224 (0.945): coatnet_rmlp_2_rw_224, RAPTOR_CROP_FRAC=0.8. Both use blendjev labels, 16 epochs and seed 42. The exact argv is in RUNS.md.
  
  - SWA = top-3 epochs ranked on Gold-58 (384: 14/11/12; 224: 15/13/12). This is where Gold touched the CoAtNet weights, so in your rerun select inside the inner fold instead.
  
  - OAI linkage manifest: we can't post it, because it is subject-level OAI data. It is filed as a restricted manifest with our NDA Study #3407 (https://nda.nih.gov/study.html?id=3407, DOI 10.15154/0hcg-f676). Anyone with an approved NDA OAI Data Use Certification can request it through NDA.
  
  If any hash doesn't match on your side, post it here and I'll check.