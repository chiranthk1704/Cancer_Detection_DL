# Histopathological Cancer Classification (Colon & Prostate)

## Overview
This repository contains implementations and experimental evaluations of multiple deep learning architectures for histopathological cancer classification on **colon cancer** and **prostate cancer** datasets.  

Several established CNN models (e.g., VGG16, ResNet50, RCCGNet, TurkerNet, ProsGradNet) were trained and benchmarked under a unified pipeline.  

The goal of this work is to develop a **proposed lightweight CNN model** that outperforms state-of-the-art architectures — particularly **ProsGradNet** [[1]](https://doi.org/10.1016/j.bspc.2025.107626) and **DRDANet** — on both colon and prostate cancer classification tasks.

ProsGradNet introduces the **Context Guided Shared Channel Residual (CGSCR) block**, which improves feature representation by structured channel grouping and contextual attention, leading to superior histopathology classification performance [[1]](https://doi.org/10.1016/j.bspc.2025.107626).

---

## Implemented Models
The following architectures were implemented and trained:

- VGG16 [[7]](https://arxiv.org/abs/1409.1556)  
- ResNet50 [[8]](https://arxiv.org/abs/1512.03385)  
- GoogLeNet (Inception) [[9]](https://arxiv.org/abs/1409.4842)  
- RCCGNet [[2]](https://www.nature.com/articles/s41598-023-31275-7)  
- LiverNet [[4]](https://doi.org/10.1007/s11548-021-02410-4)  
- BCHNet [[5]](https://doi.org/10.1371/journal.pone.0214587)  
- DRDANet  
- MDFF [[6]](https://doi.org/10.1016/j.compbiomed.2023.107385)  
- TurkerNet [[3]](https://doi.org/10.1016/j.asoc.2024.11179)  
- ProsGradNet [[1]](https://doi.org/10.1016/j.bspc.2025.107626)  

---

## Datasets

We evaluate all models on publicly available histopathological tissue datasets for **colorectal** and **prostate** cancer classification obtained from the DAX-Net repository:  
https://github.com/QuIIL/DAX-Net  

### Colorectal Tissue Dataset
The colorectal dataset consists of histopathology tissue patches extracted from whole-slide images (WSIs) and tissue microarrays (TMAs) collected between 2006–2017 from multiple patients.  

- Training set (CT_train): 7,033 images (1024×1024)  
- Validation set (CV_validation): 1,242 images (1024×1024)  
- Test set-I (CT_test-I): 1,588 images (1024×1024)  
- Test set-II (CT_test-II): 110,170 images (1144×1144)  

All images are digitized at 40× magnification and annotated by pathologists into four tissue classes:  

- Benign (BN)  
- Well-differentiated tumor (WD)  
- Moderately differentiated tumor (MD)  
- Poorly differentiated tumor (PD)  

---

### Prostate Tissue Dataset
The prostate dataset aggregates three public cohorts from Harvard Dataverse, Gleason2019, and AGGC2022 challenges, comprising histopathology tissue patches extracted from TMAs and WSIs.

- PT_train / PT_validation / PT_test-I: 22,022 images (750×750) from 886 patients  
- PT_test-II: 17,066 images (690×690) from Gleason2019 cores  
- PT_test-III: 277,203 images (512×512) from AGGC2022 WSIs  

Images were acquired at 20×–40× magnification using multiple scanners (Aperio, NanoZoomer, Philips, Zeiss, Olympus, KFBio, Akoya), ensuring cross-scanner variability.

---

These datasets provide diverse histopathological appearances and acquisition conditions, enabling robust evaluation of CNN-based cancer classification models.

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
| **ProsGradNet** | **0.9149** | **0.9172** | **0.9185** | **0.9164** |

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
| ProsGradNet  | 0.9143 | 0.9163 | 0.9166 | 0.9161 |

---

## Research Direction
Current work focuses on designing a **novel CNN architecture** that:

- improves contextual feature extraction  
- reduces parameter complexity  
- surpasses ProsGradNet and DRDANet  
- generalizes across histopathological datasets  

---

## References

## References

[1] A. Prabhu, S. Nedungatt, S. Lal, J. Kini,  
*ProsGradNet: An effective and structured CNN approach for prostate cancer grading from histopathology images*,  
Biomedical Signal Processing and Control, 105, 107626, 2025.  
https://doi.org/10.1016/j.bspc.2025.107626  

[2] A.K. Chanchal, S. Lal, R. Kumar, J.T. Kwak, J. Kini,  
*A novel dataset and efficient deep learning framework for automated grading of renal cell carcinoma from kidney histopathology images*,  
Scientific Reports, 13(1), 2023.  
https://www.nature.com/articles/s41598-023-31275-7  

[3] T. Tuncer, P.D. Barua, I. Tuncer, S. Dogan, U.R. Acharya,  
*A lightweight deep convolutional neural network model for skin cancer image classification*,  
Applied Soft Computing, 2024.  
https://doi.org/10.1016/j.asoc.2024.11179  

[4] A.A. Aatresh, K. Alabhya, S. Lal, J. Kini, P.P. Saxena,  
*LiverNet: Efficient and robust deep learning model for automatic diagnosis of sub-types of liver hepatocellular carcinoma from H&E stained histopathology images*,  
International Journal of Computer Assisted Radiology and Surgery, 16, 1549–1563, 2021.  
https://doi.org/10.1007/s11548-021-02410-4  

[5] Y. Jiang, L. Chen, H. Zhang, X. Xiao,  
*Breast cancer histopathological image classification using convolutional neural networks with small SE-ResNet module*,  
PLoS ONE, 14(3), e0214587, 2019.  
https://doi.org/10.1371/journal.pone.0214587  

[6] C. Xu, K. Yi, N. Jiang, X. Li, M. Zhong, Y. Zhang,  
*MDFF-Net: A multi-dimensional feature fusion network for breast histopathology image classification*,  
Computers in Biology and Medicine, 165, 107385, 2023.  
https://doi.org/10.1016/j.compbiomed.2023.107385  

[7] K. Simonyan, A. Zisserman,  
*Very Deep Convolutional Networks for Large-Scale Image Recognition*,  
arXiv:1409.1556, 2015.  
https://arxiv.org/abs/1409.1556  

[8] K. He, X. Zhang, S. Ren, J. Sun,  
*Deep Residual Learning for Image Recognition*,  
arXiv:1512.03385, 2015.  
https://arxiv.org/abs/1512.03385  

[9] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov,  
D. Erhan, V. Vanhoucke, A. Rabinovich,  
*Going Deeper with Convolutions*,  
arXiv:1409.4842, 2014.  
https://arxiv.org/abs/1409.4842  

[10] DRDANet — citation to be added  

[11] W. Bulten, K. Kartasalo, P.H.C. Chen, P. Ström, H. Pinckaers,  
K. Nagpal, Y. Cai, D.F. Steiner, H. van Boven, R. Vink, et al.,  
*Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge*,  
Nature Medicine, 28(1), 154–163, 2022.  
https://www.nature.com/articles/s41591-021-01620-2  
