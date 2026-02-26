# Histopathological Cancer Classification (Colon & Prostate)

## Overview
This repository contains implementations and experimental evaluations of multiple deep learning architectures for histopathological cancer classification on **colon cancer** and **prostate cancer** datasets.  

Several established CNN models (e.g., VGG16, ResNet50, RCCGNet, TurkerNet, ProsGradNet) were trained and benchmarked under a unified pipeline.  

The goal of this work is to develop a **proposed lightweight CNN model** that outperforms state-of-the-art architectures — particularly **ProsGradNet** [1] and **DRDANet** — on both colon and prostate cancer classification tasks.

ProsGradNet introduces the **Context Guided Shared Channel Residual (CGSCR) block**, which improves feature representation by structured channel grouping and contextual attention, leading to superior histopathology classification performance [1].

---

## Implemented Models
The following architectures were implemented and trained:

- VGG16 [5]  
- ResNet50 [6]  
- GoogLeNet (Inception) [7]  
- RCCGNet [2]  
- LiverNet [4]  
- BCHNet  
- DRDANet  
- MDFF  
- TurkerNet [3]  
- ProsGradNet [1]  

---

## Datasets

- **Colon Cancer Histopathology Dataset**  
  *(add dataset link)*  

- **Prostate Cancer Histopathology Dataset**  
  *(add dataset link)*  

Reference dataset used in ProsGradNet study:  
- Prostate Gleason dataset (PANDA) [1]  
  https://github.com/shyamfec/ProsGradNet  

---

## Evaluation Metrics
Model performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

## Experimental Results

### Colon Cancer Detection Results

| Model         | Accuracy | F1 Score | Precision | Recall |
|--------------|---------:|---------:|----------:|-------:|
| VGG16        | 0.2442 | 0.2205 | 0.2225 | 0.2208 |
| ResNet50     | 0.7185 | 0.6673 | 0.7366 | 0.6815 |
| GoogLeNet    | 0.7463 | 0.7367 | 0.7712 | 0.7715 |
| RCCGNet      | 0.8662 | 0.8487 | 0.8444 | 0.8543 |
| LiverNet     | 0.8043 | 0.7814 | 0.7777 | 0.7868 |
| BCHNet       | 0.8879 | 0.8721 | 0.8691 | 0.8753 |
| DRDANet      | 0.8886 | 0.8722 | 0.8680 | 0.8771 |
| MDFF         | 0.7314 | 0.6824 | 0.7003 | 0.6758 |
| TurkerNet    | 0.8360 | 0.8148 | 0.8089 | 0.8241 |
| **ProsGradeNet** | **0.9149** | **0.9172** | **0.9185** | **0.9164** |

---

### Prostate Cancer Detection Results

| Model         | Accuracy | F1 Score | Precision | Recall |
|--------------|---------:|---------:|----------:|-------:|
| VGG16        | 0.8987 | 0.8979 | 0.8998 | 0.8985 |
| ResNet50     | 0.8693 | 0.8698 | 0.8703 | 0.8694 |
| GoogLeNet    | 0.9047 | 0.9057 | 0.9066 | 0.9052 |
| RCCGNet      | 0.9033 | 0.9036 | 0.9048 | 0.9031 |
| LiverNet     | 0.9180 | 0.9181 | 0.9187 | 0.9181 |
| BCHNet       | 0.8080 | 0.8050 | 0.8102 | 0.8074 |
| **DRDANet**  | **0.9240** | **0.9242** | **0.9250** | **0.9240** |
| MDFF         | 0.7625 | 0.7636 | 0.7687 | 0.7632 |
| TurkerNet    | 0.9015 | 0.9015 | 0.9018 | 0.9019 |
| ProsGradeNet | 0.9143 | 0.9163 | 0.9166 | 0.9161 |

---

## Research Direction
Current work focuses on designing a **novel CNN architecture** that:

- improves contextual feature extraction  
- reduces parameter complexity  
- surpasses ProsGradNet and DRDANet  
- generalizes across histopathological datasets  

---

## References

[1] A. Prabhu et al., *ProsGradNet: An effective and structured CNN approach for prostate cancer grading from histopathology images*, Biomedical Signal Processing and Control, 2025.  

[2] A.K. Chanchal et al., *RCCGNet: Deep learning framework for renal cell carcinoma classification*, Scientific Reports, 2023.  

[3] T. Tuncer et al., *TurkerNet: Lightweight CNN for cancer image classification*, Applied Soft Computing, 2024.  

[4] A.A. Aatresh et al., *LiverNet: Deep learning model for liver cancer classification*, IJCARS, 2021.  

[5] K. Simonyan, A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG16)*, ICLR, 2015.  

[6] K. He et al., *Deep Residual Learning for Image Recognition (ResNet)*, CVPR, 2016.  

[7] C. Szegedy et al., *Going Deeper with Convolutions (GoogLeNet)*, CVPR, 2015.  
