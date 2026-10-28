"""Pillar-0 BreastMRI -> Knee MRI fine-tune: checkpoint loader demo.

Model: kaggle.com/models/fabotelli/pillar0-knee-finetune
Cross-anatomy transfer study for RSNA 2026 Knee Abnormality Detection:
gold-58 macro-AUC 0.797 (frozen Pillar-0 probe 0.697; task CNNs 0.89-0.90).
Strong: Effusion 0.965, Medial OA 0.941. Weak: MCL 0.630 (ligament fine
structure is where the breast->knee domain gap concentrates).
Backbone class: github.com/YalaLab/pillar-pretrain (ECL-2.0).
"""
import os

import torch

BASE = "/kaggle/input/models"  # models mount here, nested owner/slug/fw/instance/version
ckpt_path = None
for r, _, files in os.walk(BASE):  # models subtree only — tiny
    if "pillar_ft_a.final.pt" in files:
        ckpt_path = os.path.join(r, "pillar_ft_a.final.pt")
        break
print("ckpt:", ckpt_path)
state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
print("keys:", list(state))
print("epochs trained:", state["ep"] + 1)
print("backbone tensors:", len(state["model"]))
print("head:", {k: tuple(v.shape) for k, v in state["head"].items()})
# Usage: load state["model"] into CLIPMultimodalAtlas, state["head"] into
# nn.Linear(1152, 12). Input: [D,256,256] uint8 -> /255 -> z-score ->
# trilinear [192,384,384] -> repeat 3ch -> extract_vision_feats -> head.
