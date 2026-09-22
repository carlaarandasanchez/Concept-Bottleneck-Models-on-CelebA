# Concept-Bottleneck-Models-on-CelebA
This project contains a case study evaluating the trade-offs between predictive performance and interpretability in Concept Bottleneck Models (CBM). Using the CelebA dataset and a ResNet-18 backbone, we implement and compare four model variants to classify the "Smiling" facial attribute through a set of human-understandable concepts.

The project was developed for the Neural Networks course at Universidad Carlos III de Madrid (UC3M).

---

## Project Objective

The main goal is to build and compare concept-based neural architectures that bridge the gap between predictive accuracy and model transparency:

- **Baseline (M1):** Standard end-to-end classifier mapping visual features directly to the target label.
- **Concept Predictor (M2):** Multi-label head predicting a 10-concept vector from image features.
- **Pure CBM (M3):** Joint end-to-end architecture where the final prediction is forced through the concept bottleneck, ensuring full interpretability.
- **Hybrid CBM with Side Channel (M4):** Combines a concept path with a direct feature path, with dropout applied to the side channel to study concept reliance.

---

## Methodology

The project is implemented in Python using **PyTorch** and follows a structured deep learning pipeline:

- **Dataset:** CelebA — large-scale face attribute dataset with 10 binary concept labels and a "Smiling" target. Loaded via `torchvision.datasets.CelebA` using the official train/val/test splits.
- **Preprocessing:** Images resized to 128×128 and normalized with ImageNet statistics for compatibility with the pretrained ResNet-18 backbone.
- **Class Imbalance Handling:** Several concept labels are markedly imbalanced. Binary Cross-Entropy with `pos_weight` (computed from the training split) is used to prevent the loss from being dominated by the majority class and to improve concept-level learning.
- **Backbone:** ResNet-18 pretrained on ImageNet, with the final classification layer removed, providing a shared 512-dimensional feature representation.
- **Training:** Adam optimizer with learning rate 1e-4 for 3 epochs. Best checkpoint selected by validation loss.
- **Intervention Analysis:** Soft concept-level interventions on the test set to rank each concept's influence on the final "Smiling" prediction.

---

## Results Summary

| Model | Test Accuracy | Test AUROC | Mean Concept F1 |
|---|---|---|---|
| M1 — Baseline | 0.9323 | 0.9834 | — |
| M3 — Pure CBM | 0.9146 | 0.9691 | 0.5860 |
| M2 — Concept Predictor | — | — | 0.6502 (macro-F1) |
| M4 — Hybrid CBM (p=0.0) | 0.9316 | 0.9827 | 0.6555 |

The hybrid CBM (M4) at high side-channel dropout (p=0.9) achieves the best steerability, with **Bushy Eyebrows**, **Narrow Eyes**, and **Bags Under Eyes** ranking as the most influential concepts for the Smiling prediction.

---

## Conclusion

The experiments demonstrate that concept-based models can approach baseline predictive performance while enabling transparent intermediate representations and post-hoc steerability. The hybrid CBM provides a practical compromise: it preserves strong target performance while allowing concept-level interventions. Applying `pos_weight` is not only a practical fix for class imbalance but a principled choice given that accurate concept prediction is a core requirement of the CBM framework.

---

## Repository Contents

- `CBM_Project.ipynb`: Full annotated Jupyter notebook with model implementations, training loops, dropout sweep, and intervention analysis.
- `report.pdf`: Written project report with experimental setup, quantitative results, and critical discussion.

---

## Dataset Preparation
 
The dataset is loaded via `torchvision.datasets.CelebA`, which handles the official train/validation/test splits automatically.
 
**Option A — From Google Drive (recommended):** To speed up training, copy the CelebA files from your Drive to Colab's local storage before loading. This avoids reading thousands of images directly over the Drive connection. The following files are required inside a `celeba/` directory on your Drive:
 
```
list_attr_celeba.txt
list_eval_partition.txt
identity_CelebA.txt
list_bbox_celeba.txt
list_landmarks_align_celeba.txt
img_align_celeba.zip
```
 
**Option B — Direct download via torchvision:** If you don't have the dataset locally, `torchvision.datasets.CelebA` supports `download=True`, which fetches the files directly from the source. This is slower but requires no manual setup.
 
See the notebook for the full dataset setup cell.

---

## Requirements

- Python 3.x
- PyTorch + torchvision
- Google Colab (recommended for GPU access and Drive integration)
- NumPy, Matplotlib

---

## Authors

- Carla Aranda Sánchez 
- Jorge Barcia Belinchón
- Marina Juzgado Gómez-Menor
- Iván López Anca 
