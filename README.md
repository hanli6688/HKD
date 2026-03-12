# HKD
Hybrid Knowledge Distillation for Zero-Shot Anomaly Detection
# Introduction
This is the official data link of AnomalyLVM
# datasets

Industrial Visual Anomaly Detection Datasets:

MVTec AD https://drive.google.com/file/d/12IukAqxOj497J4F0Mel-FvaONM030qwP/view?usp=drive_link

VisA https://drive.google.com/file/d/1U0MZVro5yGgaHNQ8kWb3U1a0Qlz4HiHI/view?usp=drive_link

MPDD https://drive.google.com/file/d/1cLkZs8pN8onQzfyNskeU_836JLjrtJz1/view?usp=drive_link

BTAD https://drive.google.com/file/d/19Kd8jJLxZExwiTc9__6_r_jPqkmTXt4h/view?usp=drive_link

KSDD https://drive.google.com/file/d/13UidsM1taqEAVV_JJTBiCV1D3KUBpmpj/view?usp=drive_link

DTD-Synthetic https://drive.google.com/file/d/1em51XXz5_aBNRJlJxxv3-Ed1dO9H3QgS/view?usp=drive_link

Medical Visual Anomaly Detection Datasets:

HeadCT https://drive.google.com/file/d/1ore0yCV31oLwwC--YUuTQfij-f2V32O2/view?usp=drive_link

BrainMRI https://drive.google.com/file/d/1JLYyzcPG3ULY2J_aw1SY9esNujYm9GKd/view?usp=drive_link

Br35H https://drive.google.com/file/d/1qaZ6VJDRk3Ix3oVp3NpFyTsqXLJ_JjQy/view?usp=drive_link

ISIC https://drive.google.com/file/d/1atZwmnFsz7mCsHWBZ8pkL_-Eul9bKFEx/view?usp=drive_link

ColonDB https://drive.google.com/file/d/1tjZ0o5dgzka3wf_p4ErSRJ9fcC-RJK8R/view?usp=drive_link

ClinicDB https://drive.google.com/file/d/1ciqZwMs1smSGDlwQ6tsr6YzylrqQBn9n/view?usp=drive_link

TN3K https://drive.google.com/file/d/1LuKEMhrUGwFBlGCaej46WoooH89V3O8_/view?usp=drive_link

# Content
HKD is a vision–language framework for zero-shot anomaly detection that improves semantic robustness and cross-domain generalization.

• Knowledge Distillation: Transfers semantic relevance and coexistent information from fine-tuned CLIP features, enabling the student model to learn intrinsic normal–abnormal relations without real anomaly samples.

• IDAG (Intensity-Adaptive Diffusion Anomaly Generation): Generates realistic and diverse pseudo-anomalies aligned with natural image distributions to enrich training data and enhance discrimination.

• Adaptive Text Prompts: Introduces generic normal/abnormal prompts independent of object categories, reducing domain bias in textual representations.

