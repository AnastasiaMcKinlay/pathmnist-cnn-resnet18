# PathMNIST Classification with ResNet-18  
*A Deep Learning Pipeline for Histopathology Image Classification*

## Overview  
This repository contains a deep learning workflow for training, evaluating, and tuning a convolutional neural network on the **PathMNIST** dataset from **MedMNIST2D**.  
The project includes dataset preparation, baseline training, hyperparameter search, final model evaluation, and a written report.

## Dataset  
**PathMNIST** is a 9-class colorectal tissue classification dataset consisting of RGB histology images.

- Original resolution: 3 × 224 × 224  
- MedMNIST format: 3 × 28 × 28  
- This project: images resized to **64 × 64**  
- Dataset size:  
  - Train: 89,996  
  - Validation: 10,004  
  - Test: 7,180  
  - Total: 107,180 images  

The dataset is downloaded automatically using the MedMNIST Python API.

## Model  
This project uses **ResNet-18**, a convolutional neural network known for its efficiency and stable training behavior.

- Final fully connected layer adapted to 9 output classes  
- Trained from scratch (no ImageNet pretraining)  
- Inputs normalized and resized to 64 × 64  

## Training Pipeline  

**Core components:**
- Loss: CrossEntropyLoss  
- Optimizers: Adam and SGD (tested)  
- Regularization: weight decay = 1e-4  
- Learning-rate scheduler: StepLR(step_size = 5, gamma = 0.1)  
- Early stopping: patience = 3 epochs  

The training loop alternates between training and validation phases and logs: `train_loss`, `val_loss`, `train_acc`, and `val_acc`.

### Baseline configuration
- Learning rate: 1e-3  
- Batch size: 128  
- Epochs: 10  
- Optimizer: Adam  
- Weight decay: 1e-4  
- Scheduler: StepLR(step_size = 5, gamma = 0.1)  
- Early stopping patience: 3  

Baseline result: best validation accuracy ≈ **0.9725**.

## Hyperparameter Tuning  

A subset-based hyperparameter search (20% of the training set) evaluated:

- Optimizers: Adam, SGD  
- Learning rates:  
  - Adam → 1e-3, 3e-3  
  - SGD  → 1e-2, 3e-2  
- Batch sizes: 64, 128  
- Epochs per run: 3  

**Best configuration:**
- Optimizer: Adam  
- Learning rate: 1e-3  
- Batch size: 64  

This configuration showed the most stable learning behavior and highest validation accuracy during tuning.

## Final Model Performance  

Using the best hyperparameters, the final model was trained on the full training set for 10 epochs.

Final performance on the test set:
- Test accuracy: ~0.83  
- Macro AUC: ~0.97  

These results indicate strong discriminative performance across all nine tissue classes.

## Repository Structure  

```text
pathmnist-cnn/
├── notebooks/
│   ├──  pathmnist_resnet18.ipynb
│   └──  pathmnist_resnet18.html
├── reports/
│   ├── PathMNIST_Report.docx
│   └── PathMNIST_Report.pdf
├── figures/
│   ├── loss_curve.png
│   └── accuracy_curve.png
|── src/
|   └──pathmnist_resnet18.py
└── README.md
```

## How to Run  

1. Install dependencies:

```bash
pip install torch torchvision medmnist matplotlib numpy
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open and run the main notebook:

```text
notebooks/pathmnist_resnet18.ipynb
```

## Future Work  

Potential improvements include:

- Stronger data augmentation (e.g., Color Jitter, RandAugment, MixUp)  
- Larger or pretrained networks (ResNet-34/50, DenseNet-121)  
- Class-weighted or focal loss to address class imbalance  
- Automated hyperparameter tuning with tools such as Optuna  
- Model interpretability using Grad-CAM or related methods  
