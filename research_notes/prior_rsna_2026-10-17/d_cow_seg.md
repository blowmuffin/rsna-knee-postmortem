# 3D Circle of Willis Segmentation with a Stacked U-Net 
612246 | suguuuuu | 2025-10-18T01:25 | votes 12

Introduction

I struggled with the 3D multi-class segmentation task, so this post documents my process, experiments, and key takeaways.

Background

Some participants might have succeeded with a straightforward one-stage training pipeline — if so, please share your method with me!

The following describes my workaround approach that eventually stabilized training.

Preprocessing

Due to the nature of the distributed data, the imaging range (number of slices, spatial resolution, etc.) varies significantly between cases.

To stabilize training, I added a preprocessing step to extract and crop the arterial region.

This normalization reduced variation in pixel intensity and anatomical alignment, improving segmentation accuracy.

All aneurysm segmentation was performed within the cropped arterial region.

Figure: Example of cropped area (bounding box around the provided arterial segmentation mask).

[img]

Final Approach

The final setup used a two-stage cascaded 3D U-Net architecture.

At the intermediate stage, I added an auxiliary output that predicts a unified “artery” foreground class for additional supervision.

This multi-level supervision stabilized the final multi-class aneurysm segmentation.

Architecture Overview:

- Stage 1: A large model pretrained on brain datasets (public MONAI model).

- Fixed choice to leverage pretrained weights due to limited labeled data.

- Stage 2: A smaller, scratch-trained model for detailed segmentation.

Training procedure:

- Train the Stage 1 model with binary segmentation (foreground vs. background).

- Train the Stage 2 model with multi-class segmentation (13 arteries + background).

- Use different loss weights for Stage 1 and Stage 2 outputs during training.

Figure: Model architecture overview

[img]

This two-phase (or staged fine-tuning) approach yielded stable and reproducible results.

Iterative Development Process

Here’s how the final design evolved through experiments:

- Direct multi-class training from the start

→ Result: Unstable, poor convergence, scattered false positives.

- Merged all arteries into a single “foreground” class (binary setup)

→ Result: The model learned vessel structures well.

- Separated vessel detection and class discrimination into two stages (cascade)

→ Result: Stable and consistent performance.

- Inspiration: Multi-stage refinement designs like OpenPose, which progressively refine outputs using previous predictions.

- A single-stage end-to-end setup was too unstable, so splitting into Stage 1 → Stage 2 was essential.

Ultimately, the best results came from two-stage sequential training rather than one end-to-end process.

Example Code (Model Definition)

Below is a simplified example of how I connected a pretrained MONAI segmentation model and a custom U-Net into a cascade structure.

 monai.networks.nets  UNet
 monai.bundle  load

 ():
    device = config.device
    K = (config, , )  

    
    stage1 = load(
        name=,
        source=,
        device=device,
        net_override={
            : config.in_channels,
            : ,
        },
    )

    
    stage2_in_ch = config.in_channels + 
    stage2 = UNet(
        spatial_dims=,
        in_channels=stage2_in_ch,
        out_channels=K,
        channels=(, , , ),
        strides=(, , ),
        num_res_units=,
        norm=Norm.BATCH,
    ).to(device)

     (nn.Module):
         ():
            ().__init__()
            .stage1 = s1
            .stage2 = s2

         ():
             torch.is_tensor(out):
                 out
             (out, ):
                 v  out.values():
                     torch.is_tensor(v):
                         v
             RuntimeError()

         ():
            
            logit1 = ._take_tensor(.stage1(x))  
            prob1  = F.softmax(logit1, dim=)
            fg1    = prob1[:, :, ...]                 

            
            x2 = torch.cat([x, fg1], dim=)
            logit2 = .stage2(x2)

             {
                : logit1,   
                : logit2,     
            }

    model = CascadeSeg(stage1.to(device), stage2).to(device)
     model

Notes

- Since stage1 may return a dictionary, _take_tensor extracts the first tensor it finds.

- The Stage 1 output is converted to a foreground probability map, concatenated with the input, and passed to Stage 2.

- Stage 2 is a standard U-Net.

- Both coarse (2-class) and fine (multi-class) outputs can contribute to the overall loss.

## Comments
- [Simon Beck | 2026-05-03T07:14 | v0] @sugupoko  hi, thx for this cascaded solution. is there any code that i can reproduce? i am currently interested in medical imaging domain.