# Design and Implementation of an Animal Image Classification System Based on ResNet18

**Graduate Group Project Report**

**Group Members:** [Please fill in names here]  
**Affiliation:** [Please fill in university / school / program]  
**Corresponding Author / Group Leader:** [Please fill in]  
**Date:** [Please fill in]

---

## Abstract

Accurate animal image classification is a representative computer-vision task with clear demonstration value for science education, course projects, and lightweight interactive prototypes. Rather than targeting industrial wildlife-monitoring deployment, this graduate group project focuses on building a **complete, reproducible, and explainable** recognition pipeline that non-expert users can try through desktop and web interfaces. Specifically, a ResNet18 model pretrained on ImageNet is fine-tuned on the public Animals-10 benchmark to recognize ten common animal categories. Beyond model training, we engineer a software prototype covering dataset preparation, reproducible evaluation, inference, a PyQt5 desktop client, and a Flask web service. The interfaces provide automatic recognition after image uploading, Top-3 predictions, low-confidence warnings, Grad-CAM heatmaps, category descriptions, and history management. Experiments on the held-out test set show that the fine-tuned model attains strong Top-1 / Top-3 accuracy and balanced per-class F1 scores. A controlled ablation further compares ImageNet-pretrained fine-tuning with training from scratch under a matched protocol, supporting the benefit of transfer learning. We also discuss failure modes and the limitation of single-label whole-image classification. The significance of this work lies less in proposing a new architecture than in integrating classical deep-learning methods into a deployable system for graduate-level collaborative research training.

**Keywords:** Animal Image Classification; ResNet18; Transfer Learning; Animals-10; Ablation Study; Grad-CAM; Top-K Prediction; Deep Learning System; Graduate Group Project

---

## 1. Introduction

### 1.1 Background and Motivation

Image classification remains one of the core tasks in visual recognition. With large-scale annotated datasets such as ImageNet [1] and GPU computing, convolutional neural networks (CNNs) have become the dominant paradigm for visual understanding since AlexNet [2]. Animal image classification is a convenient and pedagogically rich target domain: it involves pose variation, cluttered backgrounds, and inter-class similarity, yet the label space can be kept manageable for a group project. Typical end users in our setting are course instructors reviewing a demo, students exploring a trained model, or general users who simply upload a photo and want an immediate, understandable answer—not operators of a production ecological-monitoring system.

Traditional pipelines based on handcrafted descriptors (e.g., SIFT/HOG) and shallow classifiers often generalize poorly to unconstrained animal photographs. Deep residual networks (ResNet) [3] alleviate optimization difficulties via skip connections and remain strong visual backbones. Transfer learning further reduces data and compute cost by adapting ImageNet-pretrained representations to the Animals-10 benchmark [4]. For interactive demos, a single hard label is insufficient: Top-K alternatives, low-confidence warnings, and Grad-CAM explanations [5] help users decide when to trust the model and roughly where it looked.

**Project significance.** The value of this work is threefold: (1) **application**—a usable recognition prototype for education and lightweight content interaction; (2) **methodology**—empirical evidence that ImageNet pretraining outperforms training from scratch under a matched protocol; (3) **system/engineering**—a full loop from data preparation and evaluation to dual-interface deployment with basic interpretability. We do not claim a novel network architecture; we claim a coherent graduate research artifact that is reproducible and demonstrable.

### 1.2 Problem Statement

Given an input RGB image, the system predicts one of ten animal categories, returns a confidence score, exposes Top-3 hypotheses, and visualizes Grad-CAM evidence for the predicted class. The trained model must also be packaged as interactive desktop and web applications for non-expert users. The challenge is therefore twofold: (i) obtain reliable classification performance via transfer learning on Animals-10; and (ii) expose inference, uncertainty cues, and lightweight interpretability through a maintainable software stack suitable for course demonstration.

### 1.3 Contributions

1. We establish a reproducible fine-tuning pipeline for ResNet18 on Animals-10, including dataset organization, augmentation, seeded training, early stopping, and best-model selection based on validation accuracy.  
2. We design and implement a dual-interface recognition system (PyQt5 desktop + Flask web) with Top-3 display, low-confidence warnings, Grad-CAM heatmaps, species descriptions, and history management.  
3. We provide a dedicated evaluation script that reports Top-1 / Top-3 accuracy, macro / weighted F1, per-class metrics, a confusion matrix, and per-class F1 figures for paper-ready analysis.  
4. We conduct a pretrained vs. from-scratch ablation under a matched training protocol to quantify the contribution of ImageNet initialization.  
5. We analyze recognition behavior qualitatively, including out-of-distribution failures and the inherent limitation of single-label classification.

### 1.4 Paper Organization

Section 2 reviews related work. Section 3 describes the dataset and preprocessing protocol. Section 4 presents the model, training, inference methodology (Top-K, uncertainty, Grad-CAM), and the pretrained vs. from-scratch ablation protocol. Section 5 details system architecture and implementation. Section 6 reports main test metrics, the ablation study, qualitative analysis, and discussion. Section 7 concludes the paper and outlines future research directions.

---

## 2. Related Work

### 2.1 Deep Convolutional Architectures

Following the success of early deep CNNs, architectures such as VGG [6] and Inception (GoogLeNet) [7] further improved ImageNet recognition performance, while ResNet became a widely used residual backbone. Considering the trade-off between accuracy and latency for interactive applications, we adopt ResNet18 as the backbone of this project.

### 2.2 Transfer Learning for Domain Adaptation Tasks

Transfer learning is particularly effective when target data are moderate in scale. By initializing from ImageNet-pretrained weights and replacing the final classifier, models can rapidly adapt to new label spaces. In animal recognition scenarios, pretrained low- and mid-level filters transfer well across domains. Our approach follows this paradigm: fine-tuning ResNet18 with differential learning rates for backbone and classification head.

### 2.3 Interpretability and Confidence-Aware Interfaces

Beyond offline accuracy, practical recognition systems benefit from ranked alternatives (Top-K), confidence thresholds for uncertain cases, and visual attribution methods such as Grad-CAM. These components help users understand when to trust a prediction and which image regions drive the decision. Our prototype integrates these features into both desktop and web clients.

### 2.4 Animal Recognition Systems

Existing animal recognition studies range from species identification in camera-trap imagery to educational classification demos. Many research prototypes focus only on offline accuracy and rarely deliver complete user-facing systems. In contrast, this project emphasizes an integrated pipeline from data preparation to deployment and evaluation. Object detectors such as Faster R-CNN [8] and YOLO [9] are more appropriate for multi-instance localization; for this group objective we target image-level classification as a controlled and well-scoped research task.

---

## 3. Dataset and Preprocessing

### 3.1 Animals-10 Benchmark

Animals-10 contains approximately 28,000 animal images across ten categories. Images were collected from web sources and manually inspected, making the benchmark closer to practical noisy conditions.

| ID | Class (EN) | Class (ZH) | Display name in UI |
|----|------------|------------|--------------------|
| 0 | butterfly | 蝴蝶 | Butterfly |
| 1 | cat | 猫 | Cat |
| 2 | chicken | 鸡 | Chicken |
| 3 | cow | 牛 | Cow |
| 4 | dog | 狗 | Dog |
| 5 | elephant | 象 | Elephant |
| 6 | horse | 马 | Horse |
| 7 | sheep | 羊 | Sheep |
| 8 | spider | 蜘蛛 | Spider |
| 9 | squirrel | 松鼠 | Squirrel |

### 3.2 Data Organization and Partition Protocol

Images are organized in an ImageFolder hierarchy under `data/<class>/`. For each class, samples are randomly shuffled with a fixed seed (`SEED=42`) and partitioned into training/testing subsets at approximately 9:1. Split lists are saved as `train.txt` and `test.txt` (absolute paths + class id) for reproducibility. After regeneration on the project machine, the lists contain **23,556** training samples and **2,623** test samples.

### 3.3 Preprocessing, Normalization, and Augmentation

Images are resized to 224×224 and normalized with ImageNet mean/std statistics. Training augmentation includes random horizontal flipping and color jittering (brightness / contrast / saturation); validation and inference transforms are deterministic.

---

## 4. Methodology

### 4.1 Model Formulation

Let the input RGB image be *x*, and its ground-truth class label be

*y* ∈ {0, 1, …, 9}.

The classifier is a parameterized network *fθ*(·). It maps *x* to a 10-dimensional logit vector

*z* = *fθ*(*x*) ∈ ℝ¹⁰,

where *θ* denotes all trainable parameters. Class probabilities are obtained by

*p* = softmax(*z*),  *p* ∈ ℝ¹⁰,  Σᵢ *pᵢ* = 1.

The predicted label is

*ŷ* = argmaxᵢ *pᵢ*.

Training minimizes the cross-entropy loss between *p* and the one-hot encoding of *y*. We use ResNet18 initialized from ImageNet; the final fully connected layer is replaced by a 10-way linear classifier. In `train.py`, pretrained weights are loaded unless `--no-pretrained` is set; `model.fc` is then replaced by a linear layer with `NUM_CLASSES` outputs and Xavier initialization.

**【Insert code screenshot → Code Fig. 1】**  
File: `train.py`, lines ~106–115 (`weights = ...` through `loss_fn = nn.CrossEntropyLoss()`).  
**Caption:** Code Fig. 1. ResNet18 loading, classifier-head replacement, and cross-entropy loss (`train.py`).  
**Note:** ImageNet weights are used by default; the new head outputs 10 classes and is trained with cross-entropy.

### 4.2 Optimization Strategy

Training hyperparameters are centralized in `config.py`, including image size, batch size, epochs, learning rate, train/test split ratio, random seed, confidence threshold, and early-stopping patience, which improves reproducibility.

**【Insert code screenshot → Code Fig. 2】**  
File: `config.py`, lines ~75–82 (`IMG_SIZE` through `EARLY_STOP_PATIENCE`).  
**Caption:** Code Fig. 2. Global training and inference hyperparameters (`config.py`).  
**Note:** Default fine-tuning uses 30 epochs, batch size 32, and backbone learning rate 1×10⁻⁴; the low-confidence threshold at inference is 0.50.

- Optimizer: Adam  
- Learning rates: backbone 1×10⁻⁴, classification head 1×10⁻³ (10× the backbone rate)  
- Scheduler: cosine annealing; the cosine period equals the number of requested epochs  
- Default schedule: 30 epochs, batch size 32, random seed 42  
- Early stopping: stop if validation accuracy does not improve for 8 consecutive epochs  
- Checkpoint: the best validation-accuracy weights are saved to `output/resnet18_best.pth` for deployment

Data preprocessing and augmentation are implemented in `LoadData` (`utils.py`). Training uses random horizontal flip and color jitter; validation/testing use only resize and ImageNet mean/std normalization, avoiding stochasticity at evaluation time.

**【Insert code screenshot → Code Fig. 3】**  
File: `utils.py`, lines ~21–32 (`self.train_tf` through `self.val_tf`).  
**Caption:** Code Fig. 3. Training augmentation and validation preprocessing pipelines (`utils.py`).  
**Note:** Separate train/eval transforms keep test metrics reproducible and aligned with the reported results.

### 4.3 Inference: Top-K, Low Confidence, and Grad-CAM

Uploaded images are preprocessed with validation transforms and forwarded through the network. From the softmax probability vector *p*, the system returns:

1. **Top-1** class *ŷ*, its confidence *pŷ*, and a textual description;  
2. **Top-3** ranked class–confidence pairs (the three largest entries of *p*);  
3. A **low-confidence warning** when the Top-1 confidence is below 0.50;  
4. A **Grad-CAM** heatmap computed on the last residual stage of ResNet18 (layer4), highlighting spatial evidence for the predicted class.

In `predict.py`, softmax probabilities are ranked to form a Top-3 list. If the Top-1 confidence is below `CONFIDENCE_THRESHOLD` (default 0.50), `uncertain` is set to true and a low-confidence message is returned. Desktop and web clients share the same prediction backend (`predict.py`: `predict_topk` / `predict_full` / `generate_gradcam`).

**【Insert code screenshot → Code Fig. 4】**  
File: `predict.py`, lines ~91–115 (`img_tensor = preprocess...` through the acceptable-confidence message).  
**Caption:** Code Fig. 4. Top-3 inference and low-confidence decision (`predict.py`).  
**Note:** Core path is `softmax` → `topk` → threshold comparison; shared by PyQt5 and Flask UIs.

For interpretability, Grad-CAM is computed on ResNet18 `layer4`. Forward/backward hooks collect activations and gradients; back-propagation on the target-class score yields a heatmap overlaid on the input, showing where the model attends for the predicted class.

**【Insert code screenshot → Code Fig. 5】**  
File: `predict.py`, lines ~132–155 (`generate_gradcam` through `register_full_backward_hook`).  
**Caption:** Code Fig. 5. Grad-CAM hook registration on `layer4` (`predict.py`).  
**Note:** The target is the predicted (or specified) class logit; the resulting map provides spatial evidence for the decision.

### 4.4 Offline Evaluation Protocol

`evaluate.py` loads the best checkpoint, iterates over `test.txt`, and exports:

- `output/metrics.json` (Top-1 / Top-3 accuracy, macro / weighted F1, per-class precision / recall / F1 / support, classification report)  
- `output/figures/confusion_matrix.png`  
- `output/figures/per_class_f1.png`

### 4.5 Pretrained vs. From-Scratch Ablation

To isolate the effect of transfer learning, `compare.py` trains two ResNet18 models under an identical protocol:

| Setting | Initialization | Epochs | Same as the other run |
|---------|----------------|--------|------------------------|
| Pretrained | ImageNet weights (`ResNet18_Weights.DEFAULT`) | 10 | Yes |
| From-Scratch | Random / no ImageNet weights | 10 | Yes |

Shared controls: random seed 42, Adam with backbone learning rate 1×10⁻⁴ and head learning rate 1×10⁻³, cosine annealing, batch size 32, and the same `train.txt` / `test.txt` loaders. Each run reports best validation accuracy and wall-clock time. Outputs:

- `output/compare_results.json`  
- `output/figures/compare_accuracy.png`  
- Comparison checkpoints only: `resnet18_compare_pretrained_best.pth`, `resnet18_compare_scratch_best.pth` (the deployed `resnet18_best.pth` from full training is **not** overwritten)

Model construction for the ablation is handled by `build_model` in `compare.py`: ImageNet weights are loaded when `pretrained=True`; otherwise training starts without pretrained weights. The 10-class head and the rest of the training protocol remain matched, isolating the effect of initialization.

**【Insert code screenshot → Code Fig. 6】**  
File: `compare.py`, lines ~39–44 (entire `build_model`).  
**Caption:** Code Fig. 6. Pretrained vs. from-scratch model construction for ablation (`compare.py`).  
**Note:** Only initialization differs; head structure and training protocol stay the same.

Note: the ablation uses a shorter 10-epoch budget for a fair, affordable comparison; the deployed system still uses the fully fine-tuned best checkpoint evaluated in Section 6.2.

---

## 5. System Design and Implementation

### 5.1 Architecture Overview

Three layers: data/model layer, inference layer, and application layer (desktop + web). English is used as the primary UI language for both clients. The deep-learning stack is built on PyTorch [10] and TorchVision [11]; the user interfaces are implemented with PyQt5 [12] and Flask [13].

### 5.2 Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `download_dataset.py` | Acquire and reorganize Animals-10 |
| `generate_dataset.py` | Construct reproducible `train.txt` / `test.txt` splits |
| `utils.py` | Dataset abstraction and transforms |
| `seed_utils.py` | Global seeding for reproducibility |
| `train.py` | Fine-tuning, early stopping, and checkpointing |
| `evaluate.py` | Test metrics, confusion matrix, per-class F1 figures |
| `compare.py` | Pretrained vs. from-scratch ablation (matched protocol) |
| `predict.py` | Unified inference API: Top-K, uncertainty, Grad-CAM |
| `window.py` | PyQt5 desktop client (English UI) |
| `myflask.py` + `templates/` | Flask web service (English UI) |
| `config.py` | Global configs, class metadata, confidence threshold |

Modules follow the pipeline data preparation → training/evaluation → unified inference → dual UI. `config.py` holds globals; `utils.py` and `train.py` handle data and fine-tuning; `evaluate.py` / `compare.py` produce offline metrics and the ablation; `predict.py` supplies Top-3, low-confidence flags, and Grad-CAM to `window.py` (desktop) and `myflask.py` (web). The UI layer focuses on interaction and display; recognition logic is not duplicated in the clients, which reduces inconsistency between desktop and web results. Runtime screenshots of the GUIs are preferred over long UI source listings.

### 5.3 Desktop and Web Interfaces

Both clients support:

- Image upload and automatic recognition  
- Top-1 result with confidence and class description  
- Top-3 probability bars  
- Low-confidence warning when confidence < 50%  
- Grad-CAM visualization  
- Recognition history (view / delete / clear)

### 5.4 Scope and Non-Goals

This prototype is intended for course demonstration and interactive exploration. The model performs whole-image single-label classification and does not localize multiple animals in one image. Industrial camera-trap workflows, open-set rejection beyond the fixed confidence threshold, and multi-object detection are explicitly out of scope and left as future work.

---

## 6. Experiments and Discussion

### 6.1 Experimental Setup

- Software: Python, PyTorch / TorchVision, scikit-learn, matplotlib, Pillow, PyQt5, Flask  
- Data split: 23,556 train / 2,623 test (seeded, ~9:1) on Animals-10  
- Main evaluation checkpoint: `output/resnet18_best.pth` (full fine-tuning of ResNet18)  
- Ablation: `python compare.py --epochs 10` (pretrained vs. from-scratch)  
- Hardware: AMD Ryzen 7 5800H, NVIDIA GeForce RTX 3050 Laptop GPU

### 6.2 Quantitative Results (Test Set)

Results from `evaluate.py` on 2,623 test images:

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **97.03%** |
| Top-3 Accuracy | **99.24%** |
| Macro F1 | **0.9686** |
| Weighted F1 | **0.9703** |

**Per-class F1 (test set):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Butterfly | 0.99 | 0.96 | 0.97 | 212 |
| Cat | 0.97 | 0.96 | 0.96 | 167 |
| Chicken | 0.99 | 0.97 | 0.98 | 310 |
| Cow | 0.96 | 0.95 | 0.95 | 187 |
| Dog | 0.95 | 0.97 | 0.96 | 487 |
| Elephant | 0.97 | 0.97 | 0.97 | 145 |
| Horse | 0.96 | 0.97 | 0.97 | 263 |
| Sheep | 0.97 | 0.96 | 0.96 | 182 |
| Spider | 0.98 | 0.99 | 0.99 | 483 |
| Squirrel | 0.97 | 0.97 | 0.97 | 187 |

Insert figures:

- Figure: `output/figures/confusion_matrix.png`  
- Figure: `output/figures/per_class_f1.png`  
- Figure 2: training/validation curves from the ablation logs (`output/figures/training_curves.png`)  

### 6.3 Ablation: Pretrained Fine-tuning vs. From-Scratch

We compare ImageNet-pretrained fine-tuning against training from scratch with `compare.py --epochs 10`, keeping data, seed, optimizer, learning rates, and epoch count identical. Metrics below are taken from `output/compare_results.json`.

| Setting | Best Val Accuracy | Training Time |
|---------|-------------------|---------------|
| Pretrained (ImageNet init) | **96.99%** | **2688.7 s (≈44.8 min)** |
| From-Scratch | **80.75%** | **2388.6 s (≈39.8 min)** |

Insert figure: `output/figures/compare_accuracy.png`.

**Interpretation:** under the matched 10-epoch budget, pretrained fine-tuning clearly outperforms training from scratch (96.99% vs 80.75%), supporting effective ImageNet feature transfer to Animals-10. Absolute ablation numbers are lower than Section 6.2 because the ablation uses only 10 epochs, whereas the deployed model is selected from the full fine-tuning schedule.

### 6.4 Qualitative Analysis

- **Success cases:** salient animals with moderate background complexity; Top-1 confidence typically high and Grad-CAM focuses on the animal body.  
- **Ambiguous cases:** inter-class similarity (e.g., visually similar mammals); Top-3 often still contains the correct class, consistent with the high Top-3 accuracy.  
- **Low-confidence / OOD cases:** non-animal images, heavy occlusion, or tiny objects trigger the <50% warning, signaling that the user should not fully trust Top-1.  
- **Multi-animal photos:** the system returns a single global label and cannot attribute predictions to individual instances; detection-based methods would be more suitable in that setting.

### 6.5 Discussion

Within the intended scope—an educational / demo-oriented recognition prototype—transfer learning with ResNet18 is effective and efficient enough for interactive desktop and web use. Section 6.2 shows strong test accuracy for the fully fine-tuned deployed model, while Section 6.3 isolates the contribution of ImageNet initialization under controlled conditions. Top-3 predictions, a confidence threshold, and Grad-CAM improve usability and trust without changing the classifier backbone. Remaining limitations include the closed ten-class taxonomy, imperfect calibration of softmax confidence, the shorter epoch budget in the ablation, and the inability to localize multiple animals in one image. These limits also clarify future directions: broader datasets, detection-based pipelines, longer multi-seed ablations, and stronger open-set handling—if the project were extended beyond course demonstration toward field deployment.

---

## 7. Conclusion and Future Work

This graduate group project delivered a complete animal image classification system based on ResNet18 and Animals-10. Its contribution is a reproducible pipeline that combines transfer learning, dual-interface deployment, offline evaluation, a pretrained vs. from-scratch ablation, and lightweight interpretability (Top-3, low-confidence warnings, Grad-CAM)—oriented toward teaching demonstration and interactive prototyping rather than industrial monitoring. On the test set, the fully fine-tuned system reaches 97.03% Top-1 and 99.24% Top-3 accuracy; the ablation further supports ImageNet pretraining under a matched protocol. Future work may explore multi-object detection, stronger open-set rejection, longer multi-seed ablations, and model compression if real-time or edge deployment becomes a requirement.

---

## Acknowledgments

The authors thank the course instructor and all group members for discussions and collaborative development. [Optional]

---

## References

[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 248–255.  
[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2012.  
[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.  
[4] A. Corrado, “Animals-10 dataset,” Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/alessiocorrado99/animals10  
[5] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626.  
[6] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.  
[7] C. Szegedy et al., “Going deeper with convolutions,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2015, pp. 1–9.  
[8] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2015.  
[9] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779–788.  
[10] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2019.  
[11] TorchVision maintainers and contributors, “TorchVision: PyTorch’s computer vision library.” [Online]. Available: https://pytorch.org/vision/  
[12] Riverbank Computing, “PyQt5 reference guide.” [Online]. Available: https://www.riverbankcomputing.com/static/Docs/PyQt5/  
[13] Pallets Projects, “Flask documentation.” [Online]. Available: https://flask.palletsprojects.com/

---

## Appendix A. Reproducibility Commands

```bash
pip install -r requirements.txt
python download_dataset.py
python generate_dataset.py
python train.py
python evaluate.py
python compare.py --epochs 10
python window.py
python myflask.py
```

Windows CMD examples (project path may differ):

```bat
cd /d E:\Animal_Classifier
"D:\软件\Python相关\python.exe" evaluate.py
"D:\软件\Python相关\python.exe" compare.py --epochs 10
"D:\软件\Python相关\python.exe" window.py
"D:\软件\Python相关\python.exe" myflask.py
```

VS Code terminal (PowerShell):

```powershell
cd E:\Animal_Classifier
& "D:\软件\Python相关\python.exe" compare.py --epochs 10
```

`compare.py` retrains two models (pretrained + from-scratch). On CPU it can take a long time; progress lines appear after each epoch. Results: `output/compare_results.json` and `output/figures/compare_accuracy.png`.

## Appendix B. Figures to Insert

- Figure 1 System architecture  
- Figure 2 Training / validation curves (`output/figures/training_curves.png`; from ablation epoch logs)  
- Figure 3 Desktop GUI (Top-3 + Grad-CAM + low-confidence warning)  
- Figure 4 Web interface (same features)  
- Figure 5 Confusion matrix (`output/figures/confusion_matrix.png`)  
- Figure 6 Per-class F1 (`output/figures/per_class_f1.png`)  
- Figure 7 Grad-CAM success / failure examples  
- Figure 8 Pretrained vs. from-scratch comparison (`output/figures/compare_accuracy.png`)  

**Code screenshots (insert in Sections 4.1–4.5):**

- Code Fig. 1 `train.py` ~106–115 (model head + loss)  
- Code Fig. 2 `config.py` ~75–82 (hyperparameters)  
- Code Fig. 3 `utils.py` ~21–32 (train/val transforms)  
- Code Fig. 4 `predict.py` ~91–115 (Top-3 + low confidence)  
- Code Fig. 5 `predict.py` ~132–155 (Grad-CAM hooks)  
- Code Fig. 6 `compare.py` ~39–44 (`build_model`)

## Appendix C. Group Contribution Statement

| Member | Role | Main Contribution | Estimated Effort (%) |
|--------|------|-------------------|----------------------|
| [Name] | Group leader / algorithm | [Fill in] | [ ] |
| [Name] | System / GUI | [Fill in] | [ ] |
| [Name] | Web / experiments / writing | [Fill in] | [ ] |
