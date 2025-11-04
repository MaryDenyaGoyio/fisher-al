# torch_digits_softmax_online.py
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

print("Param count:", sum(p.numel() for p in model.parameters()))  # n_params = 650


# ------------------------ Laplace ------------------------
import sys
sys.path.append('/home/marydenya/Downloads/fisher-al/Laplace')
from laplace import Laplace
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

import pickle
import os

RESULT_FILE = '/home/marydenya/Downloads/fisher-al/fisher-al/results.pkl'
RECOMPUTE = True

if os.path.exists(RESULT_FILE) and not RECOMPUTE:
    with open(RESULT_FILE, 'rb') as f:
        results = pickle.load(f)
    hessian_structures = []
else:
    results = {'diag': {}, 'kron': {}, 'lowrank': {}, 'full': {}}
    hessian_structures = ['diag', 'kron', 'lowrank', 'full']

if len(hessian_structures) > 0:
    print(f"\n{'='*80}")
    print(f"[Train] n_samples = {n_samples}, batch_size = {batch_size}, n_batches = {n_batches}")
    print(f"{'='*80}\n")

    for hess_struct in hessian_structures:
        print(f"[{hess_struct.upper()}]")

        # init model
        model = nn.Linear(64, 10)
        optimizer = optim.SGD(model.parameters(), lr=1e-2)

        # lowrank backend
        if hess_struct == 'lowrank':
            try:
                from laplace.curvature.asdfghjkl import AsdfghjklHessian
                la = Laplace(model, 'classification',
                             subset_of_weights='all',
                             hessian_structure='lowrank',
                             backend=AsdfghjklHessian,
                             backend_kwargs={'low_rank': 50})  # rank를 50으로 설정
            except Exception as e:
                print(f"Error - lowrank backend error: {e}\n")
                results[hess_struct] = {'eigenvalues': [], 'trace': [], 'logdet': [], 'test_acc': []}
                continue
        else:
            la = Laplace(model, 'classification',
                         subset_of_weights='all',
                         hessian_structure=hess_struct)



        eigvals = []
        traces = []
        logdets = []
        accs = []

        for batch_idx in range(n_batches):

            # 0) get Data
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            batch_ldr = DataLoader(Subset(TensorDataset(Xtr_t, ytr_t), list(range(start_idx, end_idx))), batch_size=batch_size, shuffle=False)

            # 1) train
            model.train()
            for epoch in range(12):
                for batch_x, batch_y in batch_ldr:
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

            # 2) eval
            model.eval()
            with torch.no_grad():
                preds = model(Xte_t).argmax(dim=1)
                acc = (preds == yte_t).float().mean().item()
            accs.append(acc)

            # 3) FI
            cml_ldr = DataLoader(Subset(TensorDataset(Xtr_t, ytr_t), list(range(0, end_idx))), batch_size=batch_size, shuffle=False)
            la.fit(cml_ldr, override=True)

            # 3-1) diag FI
            if hess_struct == 'diag':
                fisher = la.posterior_precision
                eigval = fisher.detach().cpu().numpy()
                trace = eigval.sum()
                logdet = torch.log(fisher).sum().item()

            # 3-2) kron FI
            elif hess_struct == 'kron':
                fisher_decomp = la.posterior_precision
                all_eigvals = []
                for eigvals_tuple in fisher_decomp.eigenvalues:
                    if len(eigvals_tuple) == 2:
                        layer_eigvals = torch.outer(eigvals_tuple[0], eigvals_tuple[1]).flatten()
                    elif len(eigvals_tuple) == 1:
                        layer_eigvals = eigvals_tuple[0]
                    else:
                        raise ValueError(f"Unexpected eigenvalue tuple length: {len(eigvals_tuple)}")
                    all_eigvals.append(layer_eigvals)

                eigval = torch.cat(all_eigvals).detach().cpu().numpy()
                trace = eigval.sum()
                logdet = np.log(np.maximum(eigval, 1e-10)).sum()

            # 3-3) lowrank FI
            elif hess_struct == 'lowrank':
                (U, eigvals_lr), prior_prec_diag = la.posterior_precision
                eigval = eigvals_lr.detach().cpu().numpy()
                trace = eigval.sum()
                logdet = np.log(np.maximum(eigval, 1e-10)).sum()

            # 3-4) full FI
            elif hess_struct == 'full':
                fisher = la.posterior_precision
                eigval = torch.linalg.eigvalsh(fisher).detach().cpu().numpy()
                trace = torch.trace(fisher).item()
                logdet = torch.logdet(fisher).item()

            eigvals.append(eigval)
            traces.append(trace)
            logdets.append(logdet)

        # Save
        results[hess_struct] = {
            'eigenvalues': eigvals,
            'trace': traces,
            'logdet': logdets,
            'test_acc': accs
        }
        print(f"End with test acc: {accs[-1]:.4f}\n")

    with open(RESULT_FILE, 'wb') as f:
        pickle.dump(results, f)


# Plot 부분은 항상 실행
all_hess_structs = ['diag', 'kron', 'lowrank', 'full']



# ------------------------ Plot ------------------------
n_params = sum(p.numel() for p in model.parameters())

# 1-1) eigvals
fig, axes = plt.subplots(4, 1, figsize=(10, 16))
fig.suptitle('Hess eigvals', fontsize=16)

plot_idx = 0
for hess_struct in all_hess_structs:
    if hess_struct not in results:
        continue
    data = results[hess_struct]

    ax = axes[plot_idx]
    eigvals_array = np.array(data['eigenvalues'])  # (n_batches, n_params)
    plot_idx += 1

    # lowrank는 alpha를 높게
    alpha_val = 0.8 if hess_struct == 'lowrank' else 0.3

    for i in range(eigvals_array.shape[1]):
        eigval_trajectory = eigvals_array[:, i]
        color_val = np.log10(np.mean(eigval_trajectory) + 1e-10)
        ax.plot(range(1, n_batches + 1), eigval_trajectory,
                alpha=alpha_val, linewidth=0.5, c=plt.cm.coolwarm((color_val + 10) / 20))

    ax.set_xlabel('Batch (t)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'{hess_struct.upper()}: Eigenvalues (n={eigvals_array.shape[1]})')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/marydenya/Downloads/fisher-al/fisher-al/fisher_eigenvalues.png', dpi=150)


# 2) trace, logdet
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Hess trace, logdet', fontsize=16)

all_hess_structs = ['diag', 'kron', 'lowrank', 'full']

for i, val in enumerate(['trace', 'logdet']):

    ax = axes[i]
    for hess_struct in all_hess_structs:
        if hess_struct not in results:
            continue
        # eigenvalue 개수로 나누기
        n_eigvals = len(results[hess_struct]['eigenvalues'][0])
        per_eigval = np.array(results[hess_struct][val]) / n_eigvals
        ax.plot(range(1, n_batches + 1), per_eigval,
                label=f'{hess_struct.upper()} (n={n_eigvals})', linewidth=2, markersize=6)

    ax.set_xlabel('Batch (t)')
    ax.set_ylabel(f'{val} / n_eigenvalues')
    ax.set_title(f'{val} compares')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/marydenya/Downloads/fisher-al/fisher-al/fisher_comparison.png', dpi=150)