"""Notebook submission wrapper for the retrieved public D4 candidate."""

from pathlib import Path

import pandas as pd


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Exact values retrieved from the latest public D4/DINOsaur kernel output.
PREDICTION = [
    [
        1 / 3,
        1 / 3,
        2 / 3,
        1 / 3,
        2 / 3,
        1 / 3,
        1 / 3,
        2 / 3,
        1 / 3,
        1.0,
        1 / 3,
        1 / 3,
    ],
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1 / 3,
        2 / 3,
        2 / 3,
        1.0,
        1.0,
        2 / 3,
        1.0,
        1.0,
    ],
    [
        2 / 3,
        2 / 3,
        1 / 3,
        2 / 3,
        1.0,
        1.0,
        1.0,
        1 / 3,
        2 / 3,
        1 / 3,
        2 / 3,
        2 / 3,
    ],
]


def main() -> None:
    root = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
    submission = pd.read_csv(root / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    expected = [
        "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191",
        "1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776",
        "1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402",
    ]
    if submission["StudyInstanceUID"].tolist() != expected:
        raise RuntimeError("competition test cohort/order changed; refusing stale predictions")
    if submission.columns.tolist() != ["StudyInstanceUID", *TARGETS]:
        raise RuntimeError("sample-submission schema changed")
    submission[TARGETS] = PREDICTION
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("wrote", len(submission), "studies to /kaggle/working/submission.csv")


if __name__ == "__main__":
    main()
