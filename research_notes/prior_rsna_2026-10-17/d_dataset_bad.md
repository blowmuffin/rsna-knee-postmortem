# About the dataset provided by the Hosts were totally bad.
611939 | Devkumar Biswas | 2025-10-15T17:47 | votes -13

According to the dataset, there are two CSV files provided. The first one, train.csv, contains general information such as SeriesInstanceUID, Modality, PatientAge, PatientSex, Arteries (where the aneurysm is present), and a binary classification label indicating whether an aneurysm is present or not.

The second file, train_localizers.csv, includes columns such as SeriesInstanceUID, SOPInstanceUID, coordinates, and location. Here, the SOPInstanceUID corresponds to the specific DICOM file within a series where the aneurysm is supposedly present, along with its coordinate locations and labels.

My main concern is about how we can be certain of aneurysm presence in 2D DICOM images with such clarity. It is also unclear whether these annotations have been validated by radiologists. We can observe aneurysms more accurately in 3D NIfTI (.nii) files, but many of these files are missing for corresponding series IDs, and in some cases, even entire series folders are absent.

Overall, the dataset appears to be quite disorganized. A few weeks ago, a discussion on Kaggle highlighted that the data is not correctly mapped between aneurysm presence and coordinate locations. It was also mentioned that manually annotating this dataset would require hundreds of hours of expert effort.

As quoted in that discussion:

“This project was very ambitious from the start. Not only is it one of the largest ever medical imaging competition datasets ever released, but it also has several other complexities including inhomogeneous multi-site data, nuances in DICOM format, and several different data types (image, categorical, location, segmentation, etc.). It also required hundreds of hours of expert manual annotation. We are proud of the dataset we have created, but we know it still has many flaws.”

Reference: Kaggle Discussion – https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/610101

I have trained three models so far:

YOLOv8 using CTA data

YOLOv8 using all modalities for prediction

Faster R-CNN using all modalities for classification

VA-UNet for segmentation

However, all models have only achieved leaderboard scores in the range of 0.50–0.53. I would greatly appreciate it if you could look into this issue or provide guidance on how to improve the results.

## Comments