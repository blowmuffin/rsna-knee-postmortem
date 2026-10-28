# 4th place solution
539443 | tattaka | 2024-10-09T00:21 | votes 76

Congrats to all prize and medal winners! This year's RSNA competition required us to carefully handle data and build a pipeline, which was a lot of fun. We share our solution.

Summary

Our solution detects the keypoint, which is the region of interest in the symptom, and builds a classification model using the surrounding crops as input.

The results of each model are refined by the stacking model and submitted as the final result.

[img]

Keypoint detection Model

Disk Level detection model and Keypoint detection for Axial, Sagittal T1/T2 (@yu4u)

Resize each axial slice to 128×128 and use a 2.5DCNN + LSTM model to estimate which level (L1, L2, …, S1) each slice belongs to. Subsequently, detect the boundary slices between each level. From these slices (up to five), use a UNet model to detect the left and right keypoints.

Similarly, resize the sagittal T1 slices to 128×128 and use a 2.5D CNN model to identify the left and right slices belonging to the foraminal zone where keypoints should be detected. Then, individually detect keypoints for five levels from these left and right slices.

For sagittal T2/STIR, simply extract the middle slice of the series and detect keypoints for the five levels.

Keypoint detection model for Sagittal (@tattaka)

We resized each of the Sagittal T1, T2/STIR images to 20x256x256 and predicted the xy coordinates of the keypoints.

The xy coordinates of the keypoints were taken from the shared Lumbar Coordinate Dataset .

For the backbone, we used caformer_s18, convnext_tiny, resnetrs50, and swinv2_tiny, and applied SCSE attention to the UNet Decoder.

As for the loss function, we used BCELoss * 0.2 + DICELoss * 0.8.

Classification model

Multi-view input, multi-condition output model (@tattaka)

[img]

We crop the areas around the keypoints inferred from the volumes of Sagittal T1, Sagittal T2/STIR, and Axial T2, and use them as inputs to classify the conditions at each level.

Each image is cropped to a size that is twice the distance between the neighboring keypoints. For sagittal images, padding is applied if all slices are fewer than 30, and linear interpolation is used if there are more. (There is also a model variation that simply uses linear interpolation to resize to 20.)

For axial images, slices are taken from the range of ±2 around the predicted gaps between the discs.

After the images are input into a 2D model backbone, features are extracted from each slice, and the final output is obtained using a transformer encoder and attention pooling on the extracted features. 

The model used for the final submission includes variations such as:

- A model that processes cropped slices with one or two backbones,

- Multiple augmentation patterns,

- Various preprocessing patterns for multiple slices.

The backbones used were caformer_s18, resnetrs50, rdnet_tiny , and maxxvitv2_nano.

A key technique to successfully train this model is to apply attention pooling to the features before inputting them into the transformer and calculate the auxiliary loss.

Single-view input, single-condition output model (@yu4u)

[img]

In this part, severity score is estimated via cropped images.

For Sagittal T1 and Sagittal T2/STIR images, the cropping scale is determined based on the average distance between the keypoints of the five levels, and patches are cropped centered on the keypoints. For Axial T2 images, an affine transformation is applied to position the left and right keypoints at specific locations within the patch before cropping. Subsequently, the cropped images are input into a 2.5D CNN model to calculate the severity score. For Axial T2 images, the model is used not only to predict subarticular stenosis but also spinal canal stenosis. Other combinations did not yield significant results.

Ensemble and stacking

Nelder-Mead guided stacking MLP

We constructed a stacking model using an MLP.

The key feature of this model is that, in addition to the standard skip connection, the output optimized by the Nelder-Mead method is added to the model output. (In other words, the model learns the difference between the ground truth and the Nelder-Mead results.)

The inputs for ss, scs, and any consist only of the results from their respective classification models, while nfn is fed the concatenated outputs of scs, ss, and nfn.

Stacking LightGBM and XGBoost

The outputs of the individual models are stacked using LightGBM and XGBoost. In this stacking approach, the same model is used separately for each level, and only inputs of the same type as the output target are utilized. As a result, the input dimensions are equal to the number of models multiplied by three. Using predictions for different targets was not effective.

Source code and notebooks

source code

- tattaka's part: https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public 

- yu4u's part: https://github.com/yu4u/kaggle-rsna2024-4th

notebook

- best submission notebook: https://www.kaggle.com/code/ren4yu/rsna2024-ensemble-submission-stacking-tattakav2/notebook?scriptVersionId=200046802

- MLP stacking for scs: https://www.kaggle.com/code/tattaka/rsna2024-stacking

- MLP stacking for nfn: https://www.kaggle.com/code/tattaka/rsna2024-stacking-nfn

- MLP stacking for ss: https://www.kaggle.com/code/tattaka/rsna2024-stacking-nfn

## Comments
- [kaerururu | 2024-10-15T04:40 | v1] @tattaka  @ren4yu 
  
  Congrats for prize and Share your solution ! 🎉
  
  We can't have access to your MLP stacking codes. Please check access setting 🙏
  - [tattaka | 2024-10-15T13:16 | v0] Thank you, I have changed the permissions of the notebook for MLP stacking to public.
- [Ivan Vybornov | 2024-10-09T09:57 | v2] @tattaka  Thanks for sharing! Your Multi-view input, multi-condition output is truly inspiring 🧠🧠🧠
  
  How hard is it to make your multi-view, multi-condition architecture up and running? How many iterations on the architecture/experiments worth of time does it take? And do you recommend less experienced kagglers to try and construct something similar or do you think it requires much more intuiton about the cv domain and thorough knowledge of valid computer vision achitectures and it would be better to focus on careful data processing and other basic things rather than assembling such "monsters"?
  - [tattaka | 2024-10-10T02:38 | v3] How hard is it to make your multi-view, multi-condition architecture up and running?   
    
    In my setup, the model was sensitive to hyperparameters and did not work with all backbones.
    
      How many iterations on the architecture/experiments worth of time does it take?
    
    Experiments with multi-view architectures took at least 6 hours to complete a 5-fold experiment. Most of the time, I validated the effectiveness using only 1 fold.
    
      And do you recommend less experienced kagglers to try and construct something similar or do you think it requires much more intuiton about the cv domain and thorough knowledge of valid computer vision achitectures and it would be better to focus on careful data processing and other basic things rather than assembling such "monsters"?
    
    This is just my personal opinion, but I recommend starting by building a simple model or pipeline and checking if it works. You can then use this pipeline as a starting point for experiments. By testing hypotheses and exploring things that go against your intuition, you can improve the pipeline.
- [Harshit Sheoran | 2024-10-09T00:43 | v2] Congratulations @tattaka  @ren4yu , the solution is very elegant, I liked your use of so many heads for study-level predictions to ensemble the approaches, what was your OOF CV and did it follow the LB well?
  - [tattaka | 2024-10-09T00:59 | v3] The final CV scores were:
    
     score: .
     score: .
     score: .
     score: .
    
    In our solution, the correlation between CV and LB was good.
    - [Harshit Sheoran | 2024-10-09T01:16 | v0] My CV is like this (mine alone, ensembled is 0.380):
      
      (,
       {: ,
        : ,
        : ,
        : })
      
      Could you double check my OOF submission file by calculating the score on your function, thank you!
      
      For us 0.446 CV was 0.39 LB and 0.380 CV was 0.4 LB
      
      https://www.kaggle.com/datasets/harshitsheoran/rsna-2024-oof
    - [tattaka | 2024-10-09T01:38 | v0] When you calculate your score using our notebook  , you get
      
      "Score (0.38779225210879453, {: 0.251212030446364, : 0.4966134331970362, : 0.5588322928130882, : 0.24451125197868967})".
      
      There are some slight differences, but it seems that the oof score is not incorrect.
    - [Harshit Sheoran | 2024-10-09T01:41 | v0] Thank you, so, it was just us missing something huge on subarticular, and LB not following the rest, its stupidly frustrating
    - [Quan Vu | 2024-10-09T02:38 | v1] My team single model CV: 0.4, subarticular: 0.57, spinal: 0.27, foraminal: 0.51 🥲
- [homiecal | 2024-11-03T05:37 | v0] @tattaka  @ren4yu  may I ask with the classification and keypoint detection models, do you use pre-trained weights in the backbones? or do you train from scratch?
  
  What is the typical approach when training models in a different domain, i.e. medical data MRIs, when the weights were trained on ImageNet?
- [Darwin Berrio | 2024-10-17T02:40 | v0] Congratulations on this impressive development! I would like to know if you think that using a classic machine learning model could have achieved similar performance to your solution, and in which cases you consider it better to use traditional machine learning models versus deep learning models.
- [Humayra Khanom Rime | 2024-10-12T05:41 | v0] Great! Congratulations. I haven't participated in any competition yet. Can you please share some tips so that I can do well?
- [Bartley | 2024-10-11T23:08 | v0] Interesting use of "Nelder-Mead guided stacking". Did you compare the performance between this method and a single MLP head?
  - [tattaka | 2024-10-12T02:26 | v1] It deteriorates by 0.01 compared to a single MLP head (worse than the simple Nelder-Mead).
- [MrSimple | 2024-10-11T12:15 | v0] Thanks for sharing! Your Multi-view input, multi-condition output is truly inspiring !!!!
- [C R Suthikshn Kumar | 2024-10-09T15:59 | v0] Congratulations on winning the 4th prize in this competition. Thanks for sharing the explanation about your approach with good diagrams.
- [None | 2024-10-13T22:51 | v0] 
- [None | 2024-10-09T17:06 | v0] 
- [None | 2024-10-09T01:54 | v0] 
- [Abdullah Al-Shami | 2024-10-09T08:15 | v0] thank you so much  for sharing
- [Deepak | 2024-10-09T07:35 | v0] Thank you for sharing.Helpful