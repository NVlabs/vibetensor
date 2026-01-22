#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple MNIST training with a 2-layer MLP using PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import struct
import time
import numpy as np
import os
import urllib.request
from pathlib import Path

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.1
EPOCHS = 50
HIDDEN_SIZE = 256

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading (raw numpy, same as VibeTensor)
def load_mnist_images(path):
    """Load MNIST image file (IDX format)"""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0

def load_mnist_labels(path):
    """Load MNIST label file (IDX format)"""
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

data_dir = Path(__file__).parent.parent.parent / "tmp" / "data" / "MNIST" / "raw"

_MNIST_BASE_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)
_MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    req = urllib.request.Request(url, headers={"User-Agent": "vibetorch-mnist"})
    with urllib.request.urlopen(req) as r, tmp.open("wb") as out:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, out_path)


def ensure_mnist_raw(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in _MNIST_FILES:
        p = data_dir / fname
        if p.is_file() and p.stat().st_size > 0:
            continue
        last_err: Exception | None = None
        for base in _MNIST_BASE_URLS:
            try:
                _download(base + fname, p)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"Failed to download {fname}: {last_err}")


ensure_mnist_raw(data_dir)
print(f"Loading MNIST from {data_dir}...")

train_images = load_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
train_labels = load_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
test_images = load_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
test_labels = load_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
print(f"  Train: {train_images.shape}, Test: {test_images.shape}")

# 2-layer MLP model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (B, 1, 28, 28) -> (B, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(hidden_size=HIDDEN_SIZE).to(device)
print(f"Model: {model}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training loop (numpy arrays, same as VibeTensor)
def train_epoch(model, images, labels, criterion, optimizer, batch_size=128):
    model.train()
    num_samples = images.shape[0]
    num_batches = num_samples // batch_size
    
    total_loss = 0
    correct = 0
    total = 0
    
    indices = np.random.permutation(num_samples)
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_idx = indices[start:end]
        
        data = torch.from_numpy(images[batch_idx]).to(device)
        target = torch.from_numpy(labels[batch_idx]).to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}, Acc: {correct/total:.4f}")
    
    return total_loss / num_batches, 100. * correct / total

# Evaluation (numpy arrays)
def evaluate(model, images, labels, criterion, batch_size=256):
    model.eval()
    num_samples = images.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            
            data = torch.from_numpy(images[start:end]).to(device)
            target = torch.from_numpy(labels[start:end]).to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / num_batches, 100. * correct / total

# Training
print(f"\nTraining for {EPOCHS} epochs...")
print("-" * 60)

train_start = time.time()
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_images, train_labels, criterion, optimizer, BATCH_SIZE)
    test_loss, test_acc = evaluate(model, test_images, test_labels, criterion)
    
    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

train_time = time.time() - train_start

print("-" * 60)
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Training time: {train_time:.2f}s ({EPOCHS} epochs)")
print(f"Avg epoch time: {train_time/EPOCHS:.3f}s")
