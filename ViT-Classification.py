import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score
from vit_pytorch import ViT
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
healthy_folder = ''
schizophrenia_folder = ''

# Load and preprocess images
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = tf.keras.preprocessing.image.load_img(self.file_paths[idx], target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        if self.transform:
            img_array = self.transform(img_array)
        label = self.labels[idx]
        return img_array, label

transform = transforms.Compose([
    transforms.ToTensor(),
])

file_paths_healthy = [os.path.join(healthy_folder, filename) for filename in os.listdir(healthy_folder) if filename.endswith(".png")]
file_paths_schizophrenia = [os.path.join(schizophrenia_folder, filename) for filename in os.listdir(schizophrenia_folder) if filename.endswith(".png")]

X_paths = file_paths_healthy + file_paths_schizophrenia
y = [0] * len(file_paths_healthy) + [1] * len(file_paths_schizophrenia)

# Split the data into train, validation, and test sets
X_train_val_paths, X_test_paths, y_train_val, y_test = train_test_split(X_paths, y, test_size=0.15, random_state=42)
X_train_paths, X_val_paths, y_train, y_val = train_test_split(X_train_val_paths, y_train_val, test_size=0.1765, random_state=42)  # 0.15 / 0.85 = 0.1765

train_dataset = CustomDataset(X_train_paths, y_train, transform=transform)
val_dataset = CustomDataset(X_val_paths, y_val, transform=transform)
test_dataset = CustomDataset(X_test_paths, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define ViT model and move it to GPU if available
v = ViT(
    image_size=224,
    patch_size=16,
    num_classes=2,  # Adjust according to the number of classes (e.g., 2 for binary classification)
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# Define training parameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(v.parameters(), lr=0.0001)

# Training loop with tqdm and early stopping
num_epochs = 10
train_losses = []  # For storing training losses
val_losses = []    # For storing validation losses
patience = 3       # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change in validation loss to qualify as an improvement
best_val_loss = float('inf')
wait = 0           # Counter for patience

for epoch in range(num_epochs):
    v.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
        optimizer.zero_grad()
        outputs = v(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())  # Append current batch loss
        pbar.set_postfix({'loss': loss.item()})  # Update tqdm progress bar with loss value

    # Validation
    v.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            outputs = v(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

    # Early stopping
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Evaluation on test set
v.eval()
correct_test = 0
total_test = 0
test_outputs = []
test_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
        outputs = v(images)
        test_outputs.append(outputs.cpu().numpy())
        test_labels.append(labels.cpu().numpy())
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print('Accuracy on test set: %.2f %%' % test_accuracy)
