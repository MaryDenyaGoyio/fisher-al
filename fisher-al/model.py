# ------------------------ set data, model ------------------------
import numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data
digits = load_digits()
X, y = digits.data.astype(np.float32), digits.target.astype(np.int64)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

scaler = StandardScaler().fit(Xtr)
Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

Xtr_t = torch.from_numpy(Xtr)
ytr_t = torch.from_numpy(ytr)
Xte_t = torch.from_numpy(Xte)
yte_t = torch.from_numpy(yte)

batch_size = 32
n_samples = len(Xtr_t)
n_batches = (n_samples + batch_size - 1) // batch_size

# Model
model = nn.Linear(64, 10)
criterion = nn.CrossEntropyLoss()                 
optimizer = optim.SGD(model.parameters(), lr=1e-2)

print('-'*40)
print("# params:", sum(p.numel() for p in model.parameters()))  # n_params = 650
print('-'*40)