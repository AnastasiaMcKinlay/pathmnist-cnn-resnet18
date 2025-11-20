#!/usr/bin/env python
# coding: utf-8

# 
# ## 1. Setup Environment: 
# 
# Ensure the necessary packages, such as PyTorch, and MedMNIST, are installed in your Python environment. If you will be working on multiple deep learning projects at the same time, I highly recommend using PyPI (only for Python packages) or Anaconda (for managing Python, Java, C++, etc. packages) to manage the dependencies.
# Hint: To install MedMNIST, please refer to the documentation:
# https://medmnist.com/Links to an external site.
# Links to an external site.https://github.com/MedMNIST/MedMNISTLinks to an external site.
#  
# 

# In[1]:


get_ipython().run_line_magic('pip', 'install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
get_ipython().run_line_magic('pip', 'install medmnist matplotlib scikit-learn')


# In[3]:


import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from medmnist import INFO
from medmnist import PathMNIST 

from sklearn.metrics import accuracy_score, roc_auc_score


# ## 2. Load and Preprocess the MedMNIST Dataset:
# 
# Choose one of the twelve datasets from MedMNIST2D for the experiments.
# Use the MedMNIST dataset API to download and load the dataset of your choice (e.g., PathMNIST, DermaMNIST, or OrganMNIST).
# Normalize the dataset and create data loaders for the training and testing sets.
# (Optional) Implement data augmentation techniques to improve generalization.
# Hint: Refer to the MedMNIST repository documentation for preprocessing steps: https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynbLinks to an external site.
#  
# 

# In[9]:


from medmnist import INFO, PathMNIST

data = 'pathmnist'
download = True

info = INFO[data_flag]
task = info['task']          
n_channels = info['n_channels']
n_classes = len(info['label'])

print(info)
print("Task:", task)
print("Channels:", n_channels, "Num classes:", n_classes)


# In[10]:


image_size = 64  # 28, 64

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # simple normalization around 0.5; can tune if you want
    transforms.Normalize(mean=[0.5]*n_channels, std=[0.5]*n_channels)
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*n_channels, std=[0.5]*n_channels)
])

train_dataset = PathMNIST(split='train', transform=train_transform, download=download)
val_dataset   = PathMNIST(split='val',   transform=test_transform,  download=download)
test_dataset  = PathMNIST(split='test',  transform=test_transform,  download=download)

print("Train:", len(train_dataset), "Val:", len(val_dataset), "Test:", len(test_dataset))


train_labels = train_dataset.labels.squeeze()
unique, counts = np.unique(train_labels, return_counts=True)
for c, cnt in zip(unique, counts):
    print(f"class {c}: {cnt} samples")

batch_size = 128
num_workers = 2  # set 0 if issues

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}


# ## 3. Construct the Neural Network Architecture:
# 
# Choose one the classical CNNs (e.g. AlexNet, VGG16, GoogLeNet, ResNet) for the experiments.
# Define the model in PyTorch, specifying layers, activation functions, and the forward pass.
# (Optional) Adjust the architecture of NN for a better performance.
# Hint: Example code for implementation CNNs in PyTorch can be found here: https://d2l.ai/chapter_convolutional-modern/index.html (Adapt this for MedMNIST)
#  

# In[11]:


def build_resnet18(n_channels, n_classes, pretrained=False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    if n_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            n_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    return model
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_resnet18(n_channels=n_channels, n_classes=n_classes, pretrained=False)
model = model.to(device)
print(model)


# ## 4. Define Loss Function and Optimizer:
# 
# Select an appropriate loss function for the classification task (e.g., cross-entropy loss).
# Choose an optimizer like Adam or SGD and set its initial learning rate.
# Hint: You can try different loss functions and optimizers with various hyperparameters (learning rate, momentum, etc.) to improve performance.
#  

# In[12]:


base_config = {
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 10,
    "optimizer": "adam",   # or "sgd"
    "weight_decay": 1e-4,
    "use_scheduler": True,
}

def create_optimizer_and_scheduler(model, config):
    if config["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=0.9,
            weight_decay=config["weight_decay"]
        )

    if config.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    return criterion, optimizer, scheduler


# ## 5. Train the Model:
# 
# Implement the training loop, ensuring you track the loss and accuracy as the model trains.
#  
# 
# 

# In[13]:


def train_model(model, dataloaders, dataset_sizes, config, num_epochs=10, patience=3):
    since = time.time()
    criterion, optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0

    hist = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 15)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.squeeze().long().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            hist[f"{phase}_loss"].append(epoch_loss)
            hist[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy model
            if phase == "val":
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step()

        # early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed/60:.1f}m. Best val Acc: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, hist
    
config = base_config.copy()
num_epochs = config["epochs"]

model = build_resnet18(n_channels, n_classes, pretrained=False).to(device)
best_model, history = train_model(model, dataloaders, dataset_sizes, config, num_epochs=num_epochs)


# In[20]:


# save the model
torch.save(best_model.state_dict(), "pathmnist_resnet18_best.pth")


# ## 6. Evaluate the Model:
# 
# After training, evaluate the model's performance on the test dataset.
#  
# 
# 

# In[14]:


def evaluate_on_loader(model, loader):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.squeeze().long().to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    acc = accuracy_score(all_targets, all_preds)

    # multi-class macro AUC
    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class="ovr")
    except ValueError:
        auc = None

    return acc, auc, all_targets, all_preds, all_probs

test_acc, test_auc, y_true, y_pred, y_prob = evaluate_on_loader(best_model, test_loader)
print("Test accuracy:", test_acc)
print("Test AUC:", test_auc)


# ## 7. Tune Hyperparameters:
# 
# Experiment with different hyperparameters, including learning rate, batch size, number of epochs, and any regularization techniques.
# Use techniques like learning rate scheduling and early stopping to improve performance.
#  
# 
# 

# In[21]:


#load the trained model
model = build_resnet18(n_channels, n_classes, pretrained=False).to(device)
state_dict = torch.load("pathmnist_resnet18_best.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()


# In[27]:


import numpy as np
from torch.utils.data import Subset

experiments = []

# create a fixed subset of the training data for all tuning runs
subset_frac = 0.2  # e.g., use 20% of training data
subset_size = int(len(train_dataset) * subset_frac)
subset_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
train_subset = Subset(train_dataset, subset_indices)

def run_experiment(name, config):
    print(f"\n===== Experiment: {name} =====")

    train_dataset.transform = train_transform  # base transform

    # use the subset for training
    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_subset), "val": len(val_dataset)}

    model = build_resnet18(n_channels, n_classes, pretrained=False).to(device)

    best_model, hist = train_model(
        model,
        dataloaders,
        dataset_sizes,
        config,
        num_epochs=config["epochs"],  # you can keep 3 here if you want
        patience=2,
    )

    val_acc = max(hist["val_acc"])

    experiments.append({
        "name": name,
        "config": config.copy(),
        "val_acc": val_acc,
        "history": hist,
    })

    print(f"[RESULT] {name} | best val_acc={val_acc:.4f}")
    return best_model, hist

# Hyperparameter grid: optimizer, lr, batch size (total of 8 combinations)
for opt in ["adam", "sgd"]:
    if opt == "adam":
        lr_list = [1e-3, 3e-3]
    else:  # "sgd"
        lr_list = [1e-2, 3e-2]

    for lr in lr_list:
        for bs in [64, 128]:
            cfg = base_config.copy()
            cfg["optimizer"] = opt
            cfg["lr"] = lr
            cfg["batch_size"] = bs
            cfg["epochs"] = 3  # short runs on the subset

            run_experiment(
                name=f"{opt}_lr={lr}_bs={bs}",
                config=cfg,
            )

# Identify best config
sorted_exps = sorted(experiments, key=lambda exp: exp["val_acc"], reverse=True)
print("\n===== Summary (sorted by val_acc) =====")
for exp in sorted_exps:
    print(exp["name"], "val_acc=", f"{exp['val_acc']:.4f}")

best_exp = sorted_exps[0]
print("\nBest config:", best_exp["name"], "with val_acc=", best_exp["val_acc"])
best_config = best_exp["config"]


# ## 8. Visualization and Analysis:
# 
# Plot training and validation loss and performances (e.g. accuracy, AUC score, etc.) over epochs. (only for the best hyperparameters)
#  
# 
# 

# In[28]:


best = sorted_exps[0]
hist = best["history"]

epochs = range(1, len(hist["train_loss"]) + 1)

plt.figure()
plt.plot(epochs, hist["train_loss"], label="Train loss")
plt.plot(epochs, hist["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Loss curves ({best['name']})")
plt.show()

plt.figure()
plt.plot(epochs, hist["train_acc"], label="Train acc")
plt.plot(epochs, hist["val_acc"], label="Val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title(f"Accuracy curves ({best['name']})")
plt.show()


# In[34]:


# ===== Final training with best hyperparameters =====

# Using best config, but with more epochs
final_config = base_config.copy()
final_config["optimizer"] = best_config["optimizer"]
final_config["lr"] = best_config["lr"]
final_config["batch_size"] = best_config["batch_size"]
final_config["epochs"] = 10  

# Dataloaders for final training (still using train_transform)
train_dataset.transform = train_transform
train_loader = DataLoader(
    train_dataset,
    batch_size=final_config["batch_size"],
    shuffle=True,
    num_workers=num_workers,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=final_config["batch_size"],
    shuffle=False,
    num_workers=num_workers,
)
dataloaders_final = {"train": train_loader, "val": val_loader}
dataset_sizes_final = {"train": len(train_dataset), "val": len(val_dataset)}

# Train final model
final_model = build_resnet18(n_channels, n_classes, pretrained=False).to(device)
best_model, history = train_model(
    final_model,
    dataloaders_final,
    dataset_sizes_final,
    final_config,
    num_epochs=final_config["epochs"],
    patience=3,
)



# In[35]:


from medmnist import PathMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# same normalization you used for train/val
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

test_dataset = PathMNIST(
    split='test',
    download=True,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=final_config["batch_size"],
    shuffle=False,
    num_workers=num_workers,
)


# In[36]:


test_acc, test_auc, y_true, y_pred, y_prob = evaluate_on_loader(best_model, test_loader)
print("\nFinal model performance on test set:")
print("Test accuracy:", test_acc)
print("Test AUC:", test_auc)


# In[ ]:




