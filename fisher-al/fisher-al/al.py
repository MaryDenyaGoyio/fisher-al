# torch_digits_active_learning_compare.py
import numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Subset

import logging
import structlog
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baal.active.dataset import ActiveLearningDataset
from baal.active.heuristics import BALD
from baal.bayesian.dropout import patch_module
from baal.modelwrapper import ModelWrapper, TrainingArgs
from laplace import Laplace

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

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

train_set = TensorDataset(Xtr_t, ytr_t)

# Active learning settings
init_per_class = 5
query_fraction = 0.3
acq_size = 1
batch_size = 32
n_epochs = 6
n_mc_samples = 25
pool_batch = 256
n_classes = len(np.unique(ytr))

# Build initial labelled pool
rng = np.random.default_rng(0)
labelled_idx = []
for cls in range(n_classes):
    cls_indices = np.where(ytr == cls)[0]
    rng.shuffle(cls_indices)
    labelled_idx.extend(cls_indices[:init_per_class])
labelled_idx = np.array(sorted(labelled_idx))

n_train = len(train_set)
initial_count = len(labelled_idx)
n_rounds = min(int(query_fraction * n_train), n_train - initial_count)

print(f"Train set size: {n_train}")
print(f"Initial labelled samples: {initial_count}")
print(f"Active rounds (query=1): {n_rounds}")


def evaluate(model: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).argmax(dim=1)
        return (preds == yte_t).float().mean().item()


def make_dropout_model() -> nn.Module:
    net = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 10),
    )
    return patch_module(net)


def make_plain_model() -> nn.Module:
    return nn.Linear(64, 10)


def train_plain_model(indices: np.ndarray) -> tuple[nn.Module, DataLoader]:
    model = make_plain_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    subset = Subset(train_set, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model, DataLoader(subset, batch_size=batch_size, shuffle=False)


def run_bald(initial_indices: np.ndarray) -> dict[str, list[float]]:
    label_map = np.zeros(n_train, dtype=int)
    label_map[initial_indices] = 1
    active_set = ActiveLearningDataset(train_set, labelled=label_map)
    heuristic = BALD()

    acc_hist = []
    bald_mean = []
    bald_min = []
    bald_max = []

    for round_id in range(1, n_rounds + 1):
        model = make_dropout_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-2)

        args = TrainingArgs(
            optimizer=optimizer,
            criterion=criterion,
            batch_size=batch_size,
            epoch=n_epochs,
            workers=0,
            use_cuda=False,
            replicate_in_memory=True,
        )
        wrapper = ModelWrapper(model, args)
        wrapper.train_on_dataset(active_set)

        acc = evaluate(model)
        acc_hist.append(acc)
        print(f"[BALD {round_id:04d}] labelled={active_set.n_labelled:4d} acc={acc:.4f}")

        if active_set.n_unlabelled == 0:
            bald_mean.append(np.nan)
            bald_min.append(np.nan)
            bald_max.append(np.nan)
            break

        pool = active_set.pool
        probs = wrapper.predict_on_dataset(pool, iterations=n_mc_samples, verbose=False)
        scores = heuristic.get_uncertainties(probs)

        best = int(np.argmax(scores))
        score = float(scores[best])
        bald_mean.append(score)
        bald_min.append(score)
        bald_max.append(score)
        active_set.label([best])

    return {
        'acc': acc_hist,
        'bald_mean': bald_mean,
        'bald_min': bald_min,
        'bald_max': bald_max,
    }


def run_laplace(initial_indices: np.ndarray) -> dict[str, list[float]]:
    labelled_mask = np.zeros(n_train, dtype=bool)
    labelled_mask[initial_indices] = True
    pool_indices = np.where(~labelled_mask)[0]

    acc_hist = []
    trace_hist = []

    for round_id in range(1, n_rounds + 1):
        indices = np.where(labelled_mask)[0]
        model, fit_loader = train_plain_model(indices)
        acc = evaluate(model)
        acc_hist.append(acc)
        print(f"[Laplace {round_id:04d}] labelled={len(indices):4d} acc={acc:.4f}")

        if len(pool_indices) == 0:
            trace_hist.append(np.nan)
            break

        la = Laplace(model, 'classification', subset_of_weights='all', hessian_structure='diag')
        la.fit(fit_loader, override=True)

        trace_scores = []
        pool_tensor = Xtr_t[pool_indices]
        for start in range(0, len(pool_indices), pool_batch):
            stop = start + pool_batch
            xb = pool_tensor[start:stop]
            if xb.nelement() == 0:
                continue
            Js, _ = la.backend.jacobians(xb)
            f_cov = la.functional_variance(Js)
            diag = torch.diagonal(f_cov, dim1=-2, dim2=-1)
            trace_scores.append(diag.sum(dim=-1))
        trace_scores = torch.cat(trace_scores).detach().cpu().numpy()

        best_rel = int(np.argmax(trace_scores))
        trace_val = float(trace_scores[best_rel])
        trace_hist.append(trace_val)

        chosen = pool_indices[best_rel]
        labelled_mask[chosen] = True
        pool_indices = np.delete(pool_indices, best_rel)

    return {
        'acc': acc_hist,
        'trace': trace_hist,
    }


def run_random(initial_indices: np.ndarray) -> list[float]:
    labelled_mask = np.zeros(n_train, dtype=bool)
    labelled_mask[initial_indices] = True
    pool_indices = np.where(~labelled_mask)[0]

    acc_hist = []
    rng_random = np.random.default_rng(42)

    for round_id in range(1, n_rounds + 1):
        indices = np.where(labelled_mask)[0]
        model, _ = train_plain_model(indices)
        acc = evaluate(model)
        acc_hist.append(acc)
        print(f"[Random {round_id:04d}] labelled={len(indices):4d} acc={acc:.4f}")

        if len(pool_indices) == 0:
            break

        choice = rng_random.integers(len(pool_indices))
        chosen = pool_indices[choice]
        labelled_mask[chosen] = True
        pool_indices = np.delete(pool_indices, choice)

    return acc_hist


bald_results = run_bald(labelled_idx.copy())
laplace_results = run_laplace(labelled_idx.copy())
random_acc = run_random(labelled_idx.copy())

# Accuracy comparison
min_len = min(len(bald_results['acc']), len(laplace_results['acc']), len(random_acc))
round_axis = np.arange(1, min_len + 1)

plt.figure(figsize=(9, 4))
plt.plot(round_axis, bald_results['acc'][:min_len], label='BALD (MC Dropout)', linewidth=2)
plt.plot(round_axis, laplace_results['acc'][:min_len], label='Laplace Diag Trace', linewidth=2)
plt.plot(round_axis, random_acc[:min_len], label='Random', linewidth=2)
plt.xlabel('Round')
plt.ylabel('Test Accuracy')
plt.title('Active Learning Accuracy Comparison')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('al_accuracy_compare.png', dpi=150)

# BALD acquisition values
if bald_results['bald_mean']:
    axis_bald = np.arange(1, len(bald_results['bald_mean']) + 1)
    mean_arr = np.array(bald_results['bald_mean'])
    min_arr = np.array(bald_results['bald_min'])
    max_arr = np.array(bald_results['bald_max'])

    plt.figure(figsize=(8, 4))
    plt.plot(axis_bald, mean_arr, marker='o', label='BALD mean')
    plt.fill_between(axis_bald, min_arr, max_arr, alpha=0.2, label='min-max range')
    plt.xlabel('Round')
    plt.ylabel('BALD score')
    plt.title('BALD Acquisition Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('al_bald_scores.png', dpi=150)

# Laplace trace values
if laplace_results['trace']:
    axis_trace = np.arange(1, len(laplace_results['trace']) + 1)
    trace_arr = np.array(laplace_results['trace'])

    plt.figure(figsize=(8, 4))
    plt.plot(axis_trace, trace_arr, marker='o', label='Trace (diag Laplace)')
    plt.xlabel('Round')
    plt.ylabel('Trace value')
    plt.title('Laplace Trace Acquisition Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('al_trace_scores.png', dpi=150)
