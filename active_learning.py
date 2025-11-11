import argparse
import copy
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
FISHER_DIR = REPO_ROOT / "fisher-al"
OUTPUT_DIR = REPO_ROOT / "plots" / "2_active_learning" / "saves"
PLOT_ROOT = OUTPUT_DIR.parent

PACKAGE_NAME = "fisherpkg"
FIXED_SEED = 0
FIXED_PRECISION = 1e-3
FIXED_LR = 1e-2
FIXED_RANK = 100
BATCH_SIZE = 32
TRAIN_EPOCHS = 50


def _ensure_package():
    if PACKAGE_NAME in sys.modules:
        return
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(FISHER_DIR)]
    sys.modules[PACKAGE_NAME] = pkg


def load_fisher_module(module_name: str):
    _ensure_package()
    return importlib.import_module(f"{PACKAGE_NAME}.{module_name}")


def reset_parameters(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def init_model():
    fisher_model = load_fisher_module("model")
    model = copy.deepcopy(fisher_model.model)
    model.apply(reset_parameters)
    optimizer = torch.optim.SGD(model.parameters(), lr=FIXED_LR)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion, fisher_model


def evaluate_accuracy(model, features, targets, batch_size=256):
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)


def train_epochs_on_dataset(model, optimizer, criterion, dataset, epochs):
    if len(dataset) == 0:
        return
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            model.train()
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def make_subset_dataset(dataset, indices):
    return Subset(dataset, indices)


def dataloader_for(dataset):
    batch = max(1, len(dataset))
    return DataLoader(dataset, batch_size=batch, shuffle=False)


def load_histories(directory: Path):
    histories = {}
    if not directory.exists():
        return histories
    for file in directory.glob("*.json"):
        try:
            with file.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                histories[file.stem] = data
        except json.JSONDecodeError:
            continue
    return histories


def _extract_series(history, key):
    steps = []
    values = []
    for entry in history:
        val = entry.get(key)
        if val is None:
            continue
        steps.append(entry.get("step", len(steps) + 1))
        values.append(val)
    return steps, values


def _apply_step_axis(ax, steps):
    if not steps:
        return
    min_step = min(steps)
    max_step = max(steps)
    if min_step == max_step:
        ax.set_xlim(min_step - 0.5, max_step + 0.5)
        ax.set_xticks([min_step])
    else:
        ax.set_xlim(min_step, max_step)
        ax.set_xticks(range(min_step, max_step + 1))


def plot_accuracy(histories):
    if not histories:
        return
    out_path = PLOT_ROOT / "accuracy.png"
    plt.figure(figsize=(6, 4))
    all_steps = []
    for method, history in histories.items():
        steps, acc = _extract_series(history, "accuracy")
        if not steps:
            continue
        plt.plot(steps, acc, marker="o", label=method)
        all_steps.extend(steps)
    ax = plt.gca()
    _apply_step_axis(ax, all_steps)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_optimality(histories):
    if not histories:
        return
    out_path = PLOT_ROOT / "optimality.png"
    plt.figure(figsize=(10, 4))
    for idx, key in enumerate(["A_opt", "D_opt"], start=1):
        ax = plt.subplot(1, 2, idx)
        ax_steps = []
        for method, history in histories.items():
            steps, vals = _extract_series(history, key)
            if not steps:
                continue
            ax.plot(steps, vals, marker="o", label=method)
            ax_steps.extend(steps)
        ax.set_xlabel("Step")
        ax.set_ylabel(key)
        ax.set_title(f"{key} vs Step")
        ax.grid(True, alpha=0.3)
        _apply_step_axis(ax, ax_steps)
        if idx == 1:
            ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_acq_scores(histories):
    created = False
    for method, history in histories.items():
        if method == "random":
            continue
        steps = []
        means = []
        for entry in history:
            scores = entry.get("selected_scores") or []
            if not scores:
                continue
            steps.append(entry.get("step", len(steps) + 1))
            means.append(float(np.mean(scores)))
        if not steps:
            continue
        ylabel = (
            "Mean selected score (A_acq)"
            if method == "A_acq"
            else "Mean selected score (D_acq)"
        )
        title = f"{method} mean acquisition score per step"
        out_path = PLOT_ROOT / f"{method}_selected_scores.png"
        plt.figure(figsize=(6, 4))
        plt.plot(steps, means, marker="o", label=method)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        ax = plt.gca()
        _apply_step_axis(ax, steps)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        created = True
    return created


def plot_eigenvalue_trajectories(histories):
    cmap = plt.cm.get_cmap("coolwarm")
    for method, history in histories.items():
        eig_sequences = []
        steps = []
        for entry in history:
            eigvals = entry.get("eigenvalues") or []
            if not eigvals:
                continue
            eigarr = np.sort(np.asarray(eigvals, dtype=np.float64))
            eig_sequences.append(eigarr)
            steps.append(entry.get("step", len(steps) + 1))
        if not eig_sequences:
            continue
        colors = cmap(np.linspace(0, 1, len(eig_sequences)))
        method_dir = PLOT_ROOT / method
        method_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        for idx, eigarr in enumerate(eig_sequences):
            x = np.arange(1, len(eigarr) + 1)
            plt.plot(x, eigarr, color=colors[idx], alpha=0.8)
        plt.xlabel("Sorted index")
        plt.ylabel("Eigenvalue")
        plt.title(f"{method} eigenvalues per step")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(method_dir / "eigvals_steps.png", dpi=150)
        plt.close()


def plot_final_eigenvalues(histories):
    out_path = PLOT_ROOT / "eigvals_final.png"
    plt.figure(figsize=(6, 4))
    for method, history in histories.items():
        last = None
        for entry in reversed(history):
            eigvals = entry.get("eigenvalues") or []
            if eigvals:
                last = np.sort(np.asarray(eigvals, dtype=np.float64))
                break
        if last is None:
            continue
        x = np.arange(1, len(last) + 1)
        plt.plot(x, last, label=method)
    if plt.gca().has_data():
        plt.xlabel("Sorted index")
        plt.ylabel("Eigenvalue")
        plt.title("Final-step eigenvalues")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_final_eigenvalues_with_baseline(histories):
    baseline_path = REPO_ROOT / "plots" / "1_overparam" / "saves" / "eigenval.json"
    if not baseline_path.exists():
        return
    try:
        baseline_vals = np.sort(np.asarray(json.loads(baseline_path.read_text()), dtype=np.float64))
    except json.JSONDecodeError:
        return

    out_path = PLOT_ROOT / "eigvals_final_with_baseline.png"
    plt.figure(figsize=(6, 4))

    x_baseline = np.arange(1, len(baseline_vals) + 1)
    plt.plot(x_baseline, baseline_vals, label="total", linestyle="--")

    for method, history in histories.items():
        last = None
        for entry in reversed(history):
            eigvals = entry.get("eigenvalues") or []
            if eigvals:
                last = np.sort(np.asarray(eigvals, dtype=np.float64))
                break
        if last is None:
            continue
        x = np.arange(1, len(last) + 1)
        plt.plot(x, last, label=method)

    plt.xlabel("Sorted index")
    plt.ylabel("Eigenvalue")
    plt.title("Final eigenvalues vs baseline")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_plots():
    histories = load_histories(OUTPUT_DIR)
    if not histories:
        print("No histories found for plotting.")
        return
    plot_accuracy(histories)
    plot_optimality(histories)
    plot_acq_scores(histories)
    plot_eigenvalue_trajectories(histories)
    plot_final_eigenvalues(histories)
    plot_final_eigenvalues_with_baseline(histories)


def run_active_learning(strategy, args, base_dataset, test_features, test_targets):
    model, optimizer, criterion, fisher_model = init_model()
    fisher_al = load_fisher_module("fisher_al")

    rng = np.random.default_rng(FIXED_SEED + (1 if strategy == "random" else 0))

    acq_indices: list[int] = []
    pool_indices = list(range(len(base_dataset)))

    max_limit = max(1, int(len(base_dataset) * args.rate))
    history = []
    step = 0

    while len(acq_indices) < max_limit and pool_indices:
        step += 1
        acq_dataset = make_subset_dataset(base_dataset, acq_indices)
        pool_dataset = make_subset_dataset(base_dataset, pool_indices)
        acq_loader = dataloader_for(acq_dataset)

        if strategy == "A_acq":
            scores = fisher_al.A_acq(
                model,
                acq_loader,
                pool_dataset,
                val=args.val,
                apprx=args.apprx,
                rank=FIXED_RANK,
                precision=FIXED_PRECISION,
            )["A_acq"]
        elif strategy == "D_acq":
            scores = fisher_al.D_acq(
                model,
                acq_loader,
                pool_dataset,
                val=args.val,
                apprx=args.apprx,
                rank=FIXED_RANK,
                precision=FIXED_PRECISION,
            )["D_acq"]
        elif strategy == "A_inv_acq":
            scores = fisher_al.A_inv_acq(
                model,
                acq_loader,
                pool_dataset,
                val=args.val,
                apprx=args.apprx,
                rank=FIXED_RANK,
                precision=FIXED_PRECISION,
            )["A_inv_acq"]
        else:
            scores = []

        remaining = max_limit - len(acq_indices)
        if remaining <= 0:
            break
        batch_k = min(BATCH_SIZE, remaining, len(pool_indices))
        if batch_k == 0:
            break

        scores_array = np.asarray(scores) if len(scores) else None
        if strategy == "random":
            chosen_positions = rng.choice(len(pool_indices), size=batch_k, replace=False)
        else:
            if batch_k == len(pool_indices):
                top_pos = np.arange(len(pool_indices))
            else:
                top_pos = np.argpartition(-scores_array, batch_k - 1)[:batch_k]
            chosen_positions = top_pos[np.argsort(-scores_array[top_pos])]

        chosen_positions = np.array(chosen_positions, dtype=int)
        chosen_positions_list = [int(p) for p in chosen_positions.tolist()]
        chosen_globals = [int(pool_indices[pos]) for pos in chosen_positions_list]

        for idx in sorted(chosen_positions_list, reverse=True):
            pool_indices.pop(idx)

        new_subset = make_subset_dataset(base_dataset, chosen_globals)
        train_epochs_on_dataset(model, optimizer, criterion, new_subset, TRAIN_EPOCHS)
        acq_indices.extend(chosen_globals)

        eigvals_list = []
        A_opt = None
        D_opt = None
        if acq_indices:
            acq_dataset = make_subset_dataset(base_dataset, acq_indices)
            acq_loader_for_stats = dataloader_for(acq_dataset)
            opt_stats = fisher_al.optimality(
                model,
                acq_loader_for_stats,
                val=args.val,
                output=("A_opt", "D_opt"),
                apprx=args.apprx,
                rank=FIXED_RANK,
                precision=FIXED_PRECISION,
            )
            A_opt = opt_stats.get("A_opt")
            D_opt = opt_stats.get("D_opt")
            stats = fisher_al.I_theta_x(
                model,
                acq_loader_for_stats,
                each_x=False,
                val=args.val,
                output=["eigval"],
                apprx=args.apprx,
                rank=FIXED_RANK,
            )
            eigvals = np.asarray(stats["eigval"], dtype=np.float64)
            eigvals = np.maximum(eigvals, FIXED_PRECISION)
            eigvals_list = eigvals.tolist()

        A_opt = float(A_opt) if A_opt is not None else None
        D_opt = float(D_opt) if D_opt is not None else None

        test_acc = evaluate_accuracy(model, test_features, test_targets)
        history.append(
            {
                "step": step,
                "train_size": len(acq_indices),
                "selected_indices": chosen_globals,
                "selected_scores": [float(scores[pos]) for pos in chosen_positions_list]
                if len(scores)
                else [],
                "A_opt": A_opt,
                "D_opt": D_opt,
                "accuracy": test_acc,
                "acq_scores": [float(s) for s in scores] if len(scores) else [],
                "eigenvalues": eigvals_list,
            }
        )

    return history, model


def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(history, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Active Learning via FIM optimality")
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--val", type=str, default="GGN", choices=["EF", "GGN"])
    parser.add_argument("--apprx", type=str, default="diag", choices=["diag", "full"])
    parser.add_argument("--plot", action="store_true", help="Generate plots (only plots if used alone).")
    return parser.parse_args()


def main():
    args = parse_args()
    only_plot = args.plot and len(sys.argv) == 2
    do_train = not only_plot
    do_plot = args.plot

    if do_train:
        fisher_model = load_fisher_module("model")
        train_dataset = TensorDataset(
            fisher_model.Xtr_t.clone(), fisher_model.ytr_t.clone()
        )
        test_features = fisher_model.Xte_t.clone()
        test_targets = fisher_model.yte_t.clone()

        strategies = ["A_acq", "A_inv_acq", "D_acq", "random"]
        results = {}
        for strategy in strategies:
            history, final_model = run_active_learning(
                strategy, args, train_dataset, test_features, test_targets
            )
            out_path = OUTPUT_DIR / f"{strategy}.json"
            save_history(history, out_path)
            results[strategy] = str(out_path)
            torch.save(final_model.state_dict(), OUTPUT_DIR / f"{strategy}.pth")

        print(json.dumps(results, indent=2))

    if do_plot:
        generate_plots()
        print("Generated plots at", PLOT_ROOT)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
