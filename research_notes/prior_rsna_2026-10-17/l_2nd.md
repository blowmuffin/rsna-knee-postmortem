# 2nd place solution
539452 | YujiAriyasu | 2024-10-09T01:46 | votes 99

First, I would like to express my gratitude to Kaggle and RSNA for hosting this excellent competition.

I'm also grateful to my teammates who worked hard till the end.

Our solution is a simple blend of our individual predictions and small post-processing. My teammates will likely share their solutions in the replies to this post. I'll describe my solution and post-processing below.

inference code: https://www.kaggle.com/code/yujiariyasu/rsna-lumbar-spine-2nd-place-solution

Summary

My solution is an ensemble of small models. I worked separately on axial and sagittal. Additionally, I created separate models for each target.

Basically, all models predict 3 targets: ['normal_mild', 'moderate', 'severe']. I used models that treat data from different levels and left/right as the same, without considering these distinctions. In the end, I used the team's ensemble oof to remove noisy labal data and retrain the classification model.

Axial

First, I classify which slices to use for predicting each level. I used @hengck23  code for this - thank you always for your significant contributions!

Next, I estimate the regions within each image to use for severity prediction. I trained YOLOX using the provided data.

[img]

Finally, I trained classification models using ConvNeXt Small. For spinal predictions, I directly use the regions estimated by YOLOX. For non-spinal predictions, I use only the left or right half of the image, allowing me to treat left and right labels equally.

Sagittal

First, I classify slices suitable for predicting spinal and subarticular targets. I used 2.5D images and a simple Timm model.

Next, I estimate regions for each level within the images. I trained YOLOX using data shared by my teammate @brendanartley  - thank you for your excellent contribution!

[img]

Using boxes, level each level horizontally and then crop.

[img]

Finally, I perform classification using a MIL model that accepts 5 images. The backbone is ConvNeXt Small. For spinal and subarticular, I use 5 slices centered on those predicted in the 1st stage. For foraminal, I use 5 slices centered between the spinal and subarticular slices. Some models use T1/T2 in separate channels, while others use only one.

noise reduction

My teammate discovered label noise in train dataset, so we removed samples with high loss. Using our ensemble oof (CV: 0.3687), we excluded samples where the difference between the label and the predicted value was 0.8 or greater. Due to imbalanced data, we needed to apply coefficients to the moderate and severe categories. This magic improved our score by 1% on both public and private leaderboards. I came up with this idea just two days before the deadline, so I didn't have time to try various methods or coefficients. There are likely better approaches.

This is a brief overview of my solution. There are many more intricate details that I couldn't include here.

training code: https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification

Ensemble and Post-processing

I simply weighted-averaged the predictions of each member, then applied post-processing only to spinal predictions.

For each study, I multiplied the highest predicted spinal-severe value among the 5 levels by 1.25.

Again, thanks to my teammates!

## Comments
- [Ian Pan | 2024-10-09T11:39 | v24] Thank you to my teammates @yujiariyasu , @brendanartley , and @kevin0912  for a great competition. Below are more specifics regarding my part of the solution.
  
  Stage 1a - Slice Localization
  
  Foraminal
  
  I trained a CNN-transformer model taking the 3D sagittal T1 series as input and predicting the 20 targets: the distance (using ImagePositionPatient0) from the target foramen slice as specified in the ground truth coordinates for both left and right foramina at each level (10 targets) as well as a classification label for the target foramen slice (1 for the exact slice, 0.5 for adjacent slices for each side at each level (10 targets). This was optimized using a combined smooth L1 and BCE loss. During inference, I found that taking the slice with the minimum predicted distance was better than using the max classification score. 
  
  Spinal (Sagittal)
  
  Same as above, except using sagittal T2 series as input and predicting 10 targets (no left and right for spinal canal).
  
  Subarticular
  
  Using the ground truth coordinates I assigned each axial T2 slice to one of 11 labels. If the slice was positive for either left or right subarticular, then it was assigned an intervertebral level (e.g., L1/L2, L2/L3). If the slice was in between two intervertebral levels, it was assigned the level in between (e.g., L2, L3). If it was above L1 or below S1 it was assigned L1 or S1, for a total of 11 classes. I also added 2 classes for right and left subarticular, as well as 10 targets for the distance (using ImagePositionPatient2) from the target slice for each level and side. 
  
  I also used the @hengck23 's mapping code to map the predicted sagittal T2 coordinates for each spinal canal level to the axial plane. Then, for each level, I could calculate the distance of each axial slice from the predicted sagittal T2 spinal canal level. This resulted in a vector of length 5 for each slice, representing the distance of each axial slice from the predicted sagittal T2 spinal canal location along the axial z-axis for each level. I also trained a CNN-transformer model as above, except for subarticular, I concatenated this vector to the image embedding of the slice before inputting into the transformer head. I noticed that for axial T2 slices, there were several studies where the levels were offset by 1 (e.g., L1/L2 predicted as L2/L3 or vice versa). Adding in the additional information of mapped sagittal T2 coordinates improved this significantly. 
  
  For subarticular, I found it was more effective to use the slice with maximum classification score instead of minimum distance. 
  
  Spinal (Axial)
  
  Radiologists (myself included) also use axial T2 images to grade spinal stenosis. At my institution we pretty much only look at axial T2s for this. Thus I found it very helpful to include spinal canal stenosis models trained on axial T2 images. To identify the slice, I just took the average of the right and left subarticular slices (oftentimes the same or offset by 1). 
  
  Stage 1b - Keypoint Localization
  
  For foraminal, spinal (sagittal), and subarticular, I trained a regression model on 2D images using the ground truth coordinates to predict the keypoint coordinates. For foraminal and spinal (sagittal), this involved predicting each level. For subarticular, this involved predicting each side. 
  
  For many cases, the levels (for foraminal/spinal) or left/right (for subarticular) were on different slices. I just propagated the coordinates across all slices where there was a keypoint to make training easier. 
  
  For axial spinal keypoint localization, I took the average of the right and left subarticular coordinates and multiplied the y-coordinate by 1.05, since the center of the canal is typically a bit lower than the subarticular zones (assuming the image is conventionally oriented).
  
  Stage 2 - Stenosis Grading
  
  Using the ground truth coordinates, I generated crops for each target of interest. I found that training on crops generated using ground truth coordinates was slightly better than training on crops generated using predicted coordinates from my localization models. 
  
  For the most part, I generated crops with 3 image channels, where the center channel was the target slice and the 2 other channels were the adjacent slices on either side. For subarticular, I also trained a few models with 5 image channels (using 2 adjacent slices on either side instead of 1). 
  
  I also generated crops using slice offset of 1 in either direction as well as slightly offset in the xy-plane of the image, for a total of 27 crops per target ([original slice + 1 slice offset in either direction] x [original coordinates + 8 in-plane spatial offsets]). This means using 3 image channels I actually had information across 5 slices. This augmented the training data and helped account for minor errors in slice/keypoint localization. 
  
  Each crop model was trained to predict 3 classes: normal_mild, moderate, and severe. At inference, all of the crop predictions for a target were aggregated using either median or mean. This can be thought of as a form of test-time augmentation (TTA) and improved performance significantly. The loss was a sample-weighted log loss similar to the competition metric. 
  
  I trained simple 2D CNNs treating images as 3-channel images, as well as attention pooling models and MIL models (https://github.com/mahmoodlab/CLAM), using different backbones. The simple CNNs actually did the best, but the other models added diversity to the ensemble. I used a variety of backbones, including maxvit_tiny_tf_512, coatnet_1_rw_224, NFNet-0, and CSN-ResNet101 (which is actually a model for 3D data, so for this model the channel dimension was treated as a spatial dimension).
  
  This resulted in 4 groups of models: foraminal (sagittal T1), spinal (sagittal T2), subarticular (axial T2), and spinal (axial T2). Spinal was averaged using weight of 1 for sagittal and 0.5 for axial, though there was not much difference from equal weighting.
  
  My part of the model had CV of 0.3805 and LB of 0.35. I did not test my retrained models on LB after the noise reduction step described by Yuji, but I imagine it would have been 0.34.
  
  Note on handling multiple axial T2 series
  
  Many studies had multiple axial T2 series. Several axial T2 series did not have all of the levels. The subarticular slice localization model was also used to predict whether a level was present, since it predicted what level each slice was at. If a level was duplicated across two axial T2 series, then crops from both were used. For spinal axial T2 prediction, if a level was not predicted to be present, the spinal sagittal model prediction was used on its own. 
  
  Highlights
  
  I think the top 3 most important elements of my solution were:
  
  - Training and aggregating over multiple crops (especially slices)
  
  - Using both sagittal T2 and axial T2 series to predict spinal canal stenosis
  
  - Noise reduction step (not unique to my solution)
  
  Noise reduction was a last-minute attempt for us to improve our score. I noticed during the last week as I was reviewing incorrect predictions that there were a fair number of incorrect labels. However, I thought the models could learn through the noise and that removing these samples would not change anything. Yuji ended up testing the idea, which resulted in a notable improvement in our LB score - and thankfully our final models and submissions made it in time.
  - [Rohit Chaudhari | 2024-10-12T08:45 | v0] Congratulations @vaillant  ,
    
    I have few questions regarding the solution,
    
    - In you CNN-transformer model, did you use image metadata as input to transformer or not ?
    
    - What image sizes did you use ?
    
    - In Foraminal (sagittal T1) image slices you stated that minimum predicted distance was better than using the max classification score. What was the observation for Spinal (Sagittal T2) series?
    - [Ian Pan | 2024-10-14T11:21 | v3] - No image metadata was used.
      
      - Crops were all 64 x 64.
      
      - For sagittal T2 spinal, using minimum distance was very slightly better.
    - [Rasim Berke Turan | 2025-05-01T11:42 | v0] Congratulations, I am very amazed by your work!
      
      What confuses me is that the original MRI image shapes vary from 272,272 to 1024,1024, 
      
      so 64,64 would cover a different area for different sizes of images.
      
      I wonder what size of subarticular zone you have in your cropped images.
      
      Sorry if the question is trivial, I have no previous experience
- [Bartley | 2024-10-09T02:16 | v22] Some more details
  
  Thanks to my awesome teammates, I learned a lot from them and we worked hard for this result. Also, thanks to RSNA and Kaggle for hosting this competition.
  
  TLDR; My solution is a 2-stage pipeline that uses sagittal images only. Spinal and subarticular labels are inferred from sagittal T2s and foraminal labels from sagittal T1s. I use many data augmentation tricks, pseudo labeling, TTA, and more. For details, keep reading!
  
  Stage 1
  
  In this stage, I predict the location of each disc using the competition coordinates and then crop around the area of each disc. I use the coordinate dataset here , which contains corrected coordinates and additional coordinates for Sagittal T2s.
  
  The T1 and T2 pipelines are very similar in this stage. I pass the eight middle frames of each image sequence into a 2D backbone with an attention head to predict ten coordinates (two per disc). For T1s, each level has one coordinate corresponding to the right and left foraminal zones. For T2s, each level has one coordinate behind each disc and one in front of each disc. The performance of the localization models was quite strong, with predictions falling within 5 pixels of the ground truth coordinates ~99.1% of the time.
  
  I take the average distance between levels as the height. For T1s, the width is the same as the height, but for T2s, I take the average disc width. You can see an example crop in the figure below.
  
  [img]
  
  Stage 2
  
  In this stage, we classify the labels for each cropped sequence. Each sequence is treated equally, regardless of the disc level. I used 0.5*loss_with_labels + 0.5*loss_with_pseudolabels during training. Using pseudo labels was the only denoising method I tried, and in hindsight, I should have experimented with other denoising strategies sooner.
  
  For T1s, I pass the middle 24 frames into an encoder, pass the output sequence of embeddings into an LSTM layer, and then use attention to pool the embeddings into a single feature representation. During training, I apply heavy augmentations, flip the sequence order, combine left/right sides from different sequences, and apply manifold mixup to the embeddings. The idea was to try and force the model to extract information from both sides of the sequence. For T2s, the process is similar. It differs in that it uses between 5 - 16 frames and different image sizes. The differing sizes increased the diversity of predictions. Finally, nine rotations provide TTA during inference.
  
  [img]
  
  I also worked on an Axial T2 pipeline that improved my CV by 0.01-0.02, but it was not strong enough to improve our overall team CV. I have cleaned up the codebase, and released it here . Happy Kaggling 😀
  
  --
  
  Final note, although @kevin0912  did not have models in the final ensemble, his contributions/discussions helped us achieve better performance overall as a team. Thanks again to all my teammates!
  - [C R Suthikshn Kumar | 2024-10-09T16:01 | v1] Congratulations on winning the second prize in this competition. Also, thanks for sharing the details of your solution,
  - [samu2505 | 2024-10-14T06:47 | v0] According to the dataset, Sagittal T2 is only associated with Spinal Canal Stenosis, but according to 2nd stage work flow, it's Foraminal which is associated with Sagittal T1 and Axial T2; 
    
    merged_df = train_coords.merge(train_desc, how=, on=[, ])
    merged_df.groupby([])[].value_counts()
    - [Bartley | 2024-10-15T17:33 | v1] Thanks @samu2505 , just updated the figure.
- [İbrahim Öztürk | 2024-12-25T19:28 | v3] Our prediction book saves and runs in 2 minutes, everything completes, but it continues to send, it takes too long and timed out on our last try. The change of libraries was added in the figure, but the solution to the real problem was not fully understood. How can we optimize our prediction book?
- [Evanhis | 2024-10-11T16:38 | v1] congrats brother! that's actually great :-)
- [Marek Nurzynski | 2024-10-09T08:13 | v1] Congratulations, great work!
- [Tanishk Patil | 2024-10-16T05:51 | v0] Thanks for sharing your contribution and insights. Congratulatios!
- [Mingguangyi Yang | 2024-10-14T17:16 | v0] congrats! clever solutions
- [Muhammed Tausif | 2024-10-14T09:04 | v0] Thanks for sharing your contribution and insights. Nice work.
- [MrSimple | 2024-10-11T12:14 | v0] Congratss!
- [Gauthier Avite | 2024-10-10T15:21 | v0] Congratulations and thanks for sharing your insights!
- [Priyash_07 | 2024-10-10T11:37 | v0] Congratulations Yuji! Great work!
- [None | 2024-10-09T17:48 | v0] 
- [AstonJin | 2024-10-10T07:02 | v1] congratulations!
- [Abhishek | 2024-10-10T05:03 | v1] Congratulations!!
- [Muhammed Tausif | 2024-10-14T08:52 | v0] Great, thanks. @yujiariyasu
- [Humayra Khanom Rime | 2024-10-11T18:05 | v0] Congratulations!