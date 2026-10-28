# 1st place solution
540091 | NANACHI | 2024-10-12T15:01 | votes 107

First of all, I would like to express my sincere gratitude to the competition host and the Kaggle staff for organizing such a fascinating competition. I thoroughly enjoyed this competition and learned a great deal in the process!

Furthermore, I'd like to thank @hengck23  and @brendanartley . @hengck23  's discussion and notebook were the starting point for my solution, and @brendanartley  's this dataset  helped my coordinate prediction models. I was deeply impressed by their contributions to the Kaggle community.

This is my first solution write-up, so please feel free to leave any comments or suggestions for improvement!

Summary

My solution is 2 stage approach, creating test_label_coordinates.csv and predicting severity. Furthermore, I separated 1st stage into instance_number prediction and coordinate prediction. Therefore I prepared 3 type of model, instance_number prediction model, coordinate prediction model and severity prediction model. The pipeline is shown in the following figure. 

[img]

1st stage: test_label_coordinates creation

In the 1st stage, I use 2 type of models, 3D convolution model and 2D convolution model. These models are very simple, encoder + level-separated heads. 

instance_number prediction (sagittal)

In this part, I used simple 3D ConvNeXt to predict instance_number for each level. Data that is fed into models is just normalized from 0 to 1, sorted by dicom's metadata and padded 32 to depth direction to align shape. Data preprocessing is shown in the following figure (scs example). 

[img]

In training models, I trained models 2 tasks, regression and classification, and I used L1 Loss and Cross Entropy Loss respectively. In the classification task, these heads output (bs, 32) shape logits for each level. In the regression task, these heads output (bs, 3) shape vectors for each level. (bs, 3) shape vector means (x, y, z) and I used z for depth prediction, (x, y) were used auxiliary loss. In the regression task, I normalized coordinate labels 0 to 1 for stabilizing models during training. Concretely, I used label (x', y', z') = (x/width, y/height, z/32). The model architecture is shown in the following image (scs example). I implemented 3D ConvNeXt for this task (to implement 3D ConvNeXt, I referred to this repo ). 

[img]

The results of instance_number prediction models are shown in the following table (sagt2, scs). 

model/error | 
+-0 | 
+-1 | 
+-2 | 
error>+-2 | 

cls | 
71.08% | 
27.04% | 
1.43% | 
0.44% | 

reg | 
67.48% | 
30.59% | 
1.61% | 
0.31% | 

I ensembled this 2 type of predictions using median for each level (actually I used 5 fold for each task). 

coordinate prediction(sagittal)

In coordinate prediction task, I used 2d encoder + level-separated heads, almost same as instance_number regression model. Data is 3 channel image. The image is picked up using median of instance_number of L1 ~ S1. Then the data processed normalization and reshaping (512x512). Labels are (x', y') = (x/width, y/height)

for each level, same as instance_number regression, and also I used L1 loss. The model architecture is shown in the following figure. 

[img]

I used ConvNeXt-base and Efficientnet-v2-l for this task. Before I train these models, I trained these models using @brendanartley  's dataset . These pretrained models were slightly better than pretrained models that were trained using imagenet. I ensembled these predictions using mean. 

instance_number calculation and coordinate prediction (axial)

For instance_number prediction of axial, I borrowed @hengck23  's method (notebook is here ). Then I predicted coordinates of axial, same as coordinate prediction for sagittal. 

2nd stage: severity prediction

For the 2nd stage, I attempted simple 2.5D model and MIL. 2.5D model can be implemented easily, however, MIL was better than simple 2.5D at final. 

preprocessing

Cropping method

My preprocessing strategy is cropping. For example, I cropped sagt2 image for scs; 

- pick up 5 images (center is an image that was assigned instance_number)

- reshape 512x512

- crop images using the coordinate (96 pix left and 32 pix right from coordinate x, 40 pix upper and 40 pix lower from coordinate y)

After cropping an image, the image can be like the figure below (sagt2 for scs, L1/L2). 

[img]

sagt2, sagt1 and axial were cropped for each classification task. The following tables are representing cropping range from (x, y) coordinate. 

for scs

type | 
left | 
right | 
upper | 
lower | 

sagt2 | 
96 | 
32 | 
40 | 
40 | 

axial | 
96 | 
96 | 
96 | 
96 | 

Note that when I crop images from axial, I picked up left or right  subarticular stenosis coordinate randomly, and for adjusting cropping point, I added +-20 to ss coordinate x. As a result, cropping range can be like the following figure (the example is right ss coordinate x + 20). 

[img]

for nfn

type | 
left | 
right | 
upper | 
lower | 

sagt1 (both left and right) | 
96 | 
64 | 
32 | 
32 | 

axial (right) | 
144 | 
48 | 
96 | 
96 | 

axial (left) | 
48 | 
144 | 
96 | 
96 | 

for ss

type | 
left | 
right | 
upper | 
lower | 

axial (right) | 
144 | 
48 | 
96 | 
96 | 

axial (left) | 
48 | 
144 | 
96 | 
96 | 

The following image is the range of cropping axial for right subarticular stenosis. 

[img]

data augmentations

I used several augmentations like below; 

Before cropping

- random shift of coordinate x and y (-10~+10 pix)

- random shift of instance_number (-2~+2. shifting probability was decided error probability of each instance_number prediction models)

After cropping

- RandomBrightnessContrast(p=0.25)

- ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=20, p=0.5)

Especially, random shift of instance_number was crucial for robustness of error of 1st stage. 

model architecture

My model architectures are shown in following figures. 

[EDITED]  I have updated the figure illustrating the model architecture to correct an error in the previous version. aux_attn_score in the code below is fed into cross entropy loss directly. 

[img]

[img]

I used ConvNeXt-small and Efficientnet-v2-s as the encoder. After implementing Attention-based MIL, my public LB score was improved from 0.37 -> 0.35. Then, adding bi-LSTM, aux losses and ensembling improve my score from 0.35 to 0.33. bi-LSTM + Attention-based MIL was implemented like below. 

 (nn.Module):
     ():
        (LSTMMIL, ).__init__()
        .lstm = nn.LSTM(input_dim, input_dim//, num_layers=, batch_first=, dropout=, bidirectional=)
        .aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, )
        )
        .attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, )
        )
     ():
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = .lstm(bags)
        attn_scores = .attention(bags_lstm).squeeze(-)
        aux_attn_scores = .aux_attention(bags_lstm).squeeze(-)
        attn_weights = torch.softmax(attn_scores, dim=-)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(), bags_lstm).squeeze()

         weighted_instances, aux_attn_scores

what didn't work

- MAMBA and Self-Attention instead of bi-LSTM

- sharing weight between aux_attention layer and attention layer

- sagt1 image for scs, sagt1 and sagt2 image for ss, sagt2 image for nfn

- long epochs (I used 7 epochs for convnext-small and 14 epochs for efficientnet-v2-s)

- large models (convnext-large < convnext-base < convnext-small in my experiments)

- vision transformers (I think this was my problem. but convolution models were better than vits in my experiments)

code

All training code is implemented in google colaboratory. All models are used for this inference code . Following links are pairs of model name & training notebook link. You can check these model name in the inference code . 

You can train on google colaboratory environment with T4 + high memory. 

- models 

instance number prediction models (SCS)

- scs_depth_1024_ssr: notebook 

- scs_depth: notebook 

- scs_depth_1024_ssr_l1: notebook 

instance number prediction models (NFN)

- nfn_depth_1024_ssr: notebook 

- nfn_depth: notebook 

- nfn_depth_1024_ssr_l1: notebook 

coordinate prediction models (SCS)

- scs_detect_pre: notebook 

- scs_detect_pre_effv2l: notebook 

coordinate prediction models (NFN)

- nfn_detect_pre: notebook 

- nfn_detect_pre_effv2l: notebook 

coordinate prediction models (SS)

- ss_detect: notebook 

severity prediction models (SCS)

- _scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp: notebook 

- _scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_effv2s_for_exp: notebook 

severity prediction models (NFN)

- nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_convnext-s: notebook 

- nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_effv2s: notebook 

severity prediction models (SS)

- ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_effv2s: notebook 

- ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_convnext-s: notebook 

coordinate pretrained models

this notebook  is pre-training code for coordinate prediction models with this dataset . model checkpoints are in this dataset

## Comments
- [Bartley | 2024-10-12T16:01 | v4] Great first write up and congrats on the strong finish!
  
  Did the Attention-based MIL improvements improve your local CV as much as the LB? Also, do you know how much of the improvement was due to the auxillary loss?
  - [NANACHI | 2024-10-13T11:55 | v2] Thank you Bartley! Also, congratulations on winning a gold medal!
    
      Attention-based MIL improvements improve your local CV as much as the LB?
    
    Improvements of my local cv by MIL are smaller than the improvement of LB. I lost log of actual improvements of cv, but I remember the improvements are less than 0.012. However public LB improved 0.3729 -> 0.3588 (diff is 0.0141) and private LB improved 0.4259 -> 0.4062 (diff is 0.0197). 
    
      do you know how much of the improvement was due to the auxillary loss?
    
    Actually, I submitted bi-LSTM+aux loss model at the same time, so I don't have the result of the improvements on LB by aux losses. The table below is the result of validation data of scs (same seed, same preprocess, same hyper params and same architecture with/without aux head)
    
    fold | 
    without aux | 
    with aux | 
    
    0 | 
    0.254 | 
    0.240 | 
    
    1 | 
    0.283 | 
    0.254 | 
    
    2 | 
    0.267 | 
    0.254 | 
    
    3 | 
    0.264 | 
    0.252 | 
    
    4 | 
    0.244 | 
    0.261 | 
    
    mean | 
    0.2624 | 
    0.2522 |
    - [Bartley | 2024-10-13T13:08 | v2] Interesting, the improvement for aux loss is a more significant than I thought. Thanks!
- [samu2505 | 2024-10-14T05:32 | v1] In the classification task, these heads output (bs, 32) shape logits for each level.; what targets were used since there are 5 levels for each image?
- [Mingjie Wang | 2024-10-14T02:15 | v2] Congratulations! Could you explain in detail why auxiliary loss is effective? I understand it should be nearly equivalent to the original architecture.
  - [NANACHI | 2024-10-14T07:02 | v2] Thank you!
    
    In my opinion, auxiliary loss makes it possible to add context to feature vectors. For example, I added depth aux head after bi-LSTM. This is because I wanted to incorporate the context of which slice the annotators were focusing on into the feature vectors output by the LSTM. I think this allows the main attention layer to function more effectively.
- [A D Vashishta | 2026-08-10T01:37 | v0] This is very useful for my project, as i am working on similar predictive layer for Spine CSM , would like to connect with you further to seek your advice
- [tim062912 | 2025-01-24T01:58 | v0] Could you kindly provide the necessary file? Without it, we are unable to run the notebook.
- [Switch9527 | 2024-11-10T10:06 | v0] Hello! I’d like to ask about the three model notebooks in the instance number prediction models (SCS). I assume that these three models aim to achieve the same objective, with the only difference being their model structures—is that correct? Will these three models be integrated in the inference code? If my understanding is incorrect, could you please point it out? Thank you! Apologies for my slow progress with reading the code; so far, I’ve only reviewed these three notebooks.
- [homiecal | 2024-11-04T11:03 | v0] Hi @wadakoki  what hardware did you use for the competition? Was it just the free quota on Google Colab Notebooks?
- [Kenny | 2024-10-31T09:40 | v0] thanks for sharing. Great solution and explanation.
- [Malabh Bakshi | 2024-10-18T13:19 | v0] Amazing Presentation @wadakoki
- [SUZUKI SEIYA | 2024-10-18T08:28 | v0] You're number one!
- [Yisak Birhanu Bule | 2024-10-15T16:34 | v0] congra man you deserve  it
- [dragon zhang | 2024-10-14T02:26 | v0] thanks for sharing. Piece of artist solution cake.
- [Vinothkumar Sekar | 2024-10-15T14:48 | v0] Good work.!!! 👍
- [Rudrersh Asagodu | 2024-10-15T12:45 | v0] Congratulations!
- [Muhammed Tausif | 2024-10-15T07:32 | v0] Thanks for sharing your approach.
- [Usaid Ahmad | 2024-10-14T11:22 | v0] good work!
- [None | 2024-10-31T09:39 | v0] 
- [None | 2024-10-15T17:00 | v0] Congratulations! 😃