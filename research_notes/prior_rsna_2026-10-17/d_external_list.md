# External dataset list by chatGPT search
597777 | suguuuuu | 2025-08-07T11:42 | votes 16

1. Intracranial Aneurysm CTA Images and 3D Models Dataset

- Description: High-resolution CTA scans from 99 patients (105 aneurysm cases) plus 44 healthy controls. Includes 3D mesh models of aneurysms and parent vessels, as well as CFD-derived morphological and hemodynamic parameters.

- Usage Example: Pretrain a risk-prediction model by reconstructing CTA into 3D and using CFD features.

2. Time-of-Flight MRA of Intracranial Aneurysms with Interval Surveillance Imaging

- Description: TOF-MRA scans from 63 patients (24 with longitudinal follow-up). Comes with vessel and aneurysm segmentation masks, clinical annotations, 3D models, and a 3D Slicer project.

- Usage Example: Time-series analysis for aneurysm growth prediction or additional training data for segmentation models.

3. Royal Brisbane TOF-MRA Intracranial Aneurysm Database

- Description: TOF-MRA scans of 63 patients (85 aneurysm cases) with voxel-level segmentations and expert annotations.

- Usage Example: Fine-grained segmentation training using 3D CNNs or weakly supervised detection.

4. Lausanne UIA TOF-MRA Dataset (Di Noto et al., 2022)

- Description: 284 subjects (157 aneurysm cases, 127 healthy controls). Includes 246 coarse (spherical) labels and 38 detailed labels—ideal for mixed weak/strong supervision.

- Usage Example: Benchmarking weakly supervised learning frameworks.

5. ADAM Dataset for UIA Segmentation (Timmins et al.)

- Description: 113 subjects (93 unruptured intracranial aneurysm cases, 20 healthy). Provides voxel-level segmentation masks focused on central cerebral vessels.

- Usage Example: External validation set for transfer-learning and generalization studies.

6. IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning

- Description: Point-cloud and mesh representations of intracranial aneurysms with classification and part-segmentation benchmarks.

- Usage Example: Training point-cloud networks or mesh-CNNs to learn shape features.

7. Aneumo: Large-Scale Multimodal Aneurysm CFD Dataset

- Description: 427 real aneurysm shapes synthetically deformed into 10,660 models, with 85,280 CFD simulation results and segmentation masks.

- Usage Example: Benchmarking CFD parameter estimation or augmenting data synthetically.

Important:

- Only publicly available, free datasets are allowed.

- List dataset name, URL, and license in your forum post by October 14, 2025.

- Use of private or subscription-based data is prohibited.

Combining these datasets for pretraining, augmentation, and multimodal analysis can help boost the accuracy of your aneurysm detection models.

## Comments