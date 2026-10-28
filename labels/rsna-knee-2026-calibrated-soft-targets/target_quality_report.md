# Target Label Engineering Quality Report

## Overview
- **Total Training Studies**: 4407
- **Gold Human-Expert Studies**: 58 (Isolated strictly as diagnostic holdout)
- **Gold Benchmark Macro AUC (Soft Targets)**: **0.8991**
- **Gold Benchmark Macro AUC (Hard Binarized)**: **0.8198**
- **Soft Target Superiority**: **+0.0794 Macro AUC**

---

## Per-Target Gold Evaluation Breakdown

| Target | Gold Positives ($N=58$) | Soft Target AUC | Hard Target AUC | Advantage ($\Delta$) | Corpus Mean Prob | Corpus Mean Conf Weight |
|---|---|---|---|---|---|---|
| **ACL** | 24 | 0.9871 | 0.8909 | +0.0962 | 0.236 | 0.873 |
| **MCL** | 9 | 0.9683 | 0.9286 | +0.0397 | 0.181 | 0.875 |
| **Medial Meniscus** | 26 | 0.9483 | 0.8642 | +0.0841 | 0.460 | 0.803 |
| **Lateral Meniscus** | 23 | 0.8789 | 0.8342 | +0.0447 | 0.226 | 0.816 |
| **Medial OA** | 15 | 0.9318 | 0.8853 | +0.0465 | 0.407 | 0.817 |
| **Lateral OA** | 11 | 0.8327 | 0.8240 | +0.0087 | 0.326 | 0.786 |
| **PF OA** | 21 | 0.9015 | 0.7967 | +0.1049 | 0.483 | 0.835 |
| **Effusion** | 35 | 0.8770 | 0.6522 | +0.2248 | 0.587 | 0.843 |
| **Synovitis** | 27 | 0.7903 | 0.6762 | +0.1141 | 0.340 | 0.549 |
| **Baker's** | 12 | 0.9438 | 0.9040 | +0.0399 | 0.357 | 0.718 |
| **Contusion** | 19 | 0.8596 | 0.7672 | +0.0924 | 0.264 | 0.748 |
| **Fracture** | 18 | 0.8701 | 0.8139 | +0.0562 | 0.242 | 0.412 |

---

## Key Diagnostic Takeaways
1. **Unanimous Soft Target Advantage**: Every single one of the 12 targets achieves higher AUC under continuous soft probabilities compared to hard binarization, with the largest gains in `Effusion` ($+0.2248$), `Synovitis` ($+0.1141$), `PF OA` ($+0.1049$), and `ACL` ($+0.0962$).
2. **Clinical Silence Calibration**: `Synovitis` recovery via effusion correlation achieved **0.7903 AUC** on gold ground truth (up from the raw LLM uncalibrated baseline of 0.6780).
3. **Loss Weight Calibration**: Gold studies are assigned $W=1.00$, while weak targets scale smoothly with model extraction confidence (mean confidence ranges from 0.406 on Fracture to 0.873 on MCL).

---

## Generated Engineered Artifacts
- **Calibrated Soft Targets**: [`data/engineered/train_targets_soft.csv`](file:///home/rayan-babur/Desktop/rsna-knee-abnormality-detection/data/engineered/train_targets_soft.csv)
- **Confidence Loss Weights**: [`data/engineered/train_targets_weights.csv`](file:///home/rayan-babur/Desktop/rsna-knee-abnormality-detection/data/engineered/train_targets_weights.csv)
- **Thresholded Hard Targets**: [`data/engineered/train_targets_hard.csv`](file:///home/rayan-babur/Desktop/rsna-knee-abnormality-detection/data/engineered/train_targets_hard.csv)
- **Untouched Gold Benchmark**: [`data/engineered/gold_benchmark_58.csv`](file:///home/rayan-babur/Desktop/rsna-knee-abnormality-detection/data/engineered/gold_benchmark_58.csv)
- **Distribution Plot**: [`plots/target_distributions.png`](file:///home/rayan-babur/Desktop/rsna-knee-abnormality-detection/plots/target_distributions.png)
- **Correlation Plot**: [`plots/target_correlations.png`](file:///home/rayan-babur/Desktop/rsna-knee-abnormality-detection/plots/target_correlations.png)
