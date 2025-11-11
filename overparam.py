import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parent
MODEL_FILE = REPO_ROOT / "fisher-al" / "model.py"
LAPLACE_FILE = REPO_ROOT / "fisher-al" / "laplace_redux.py"
PLOT_DIR = REPO_ROOT / "plots" / "1_overparam"
SAVE_DIR = PLOT_DIR / "saves"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_module():
    return _load_module(MODEL_FILE, "fisher_model")


def load_laplace_module():
    return _load_module(LAPLACE_FILE, "laplace_redux")


def accuracy(model, features, targets):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()


def compute_eigenvalues(model, data_loader, laplace_module, curvature_type, approx_type):
    stats = laplace_module.I_theta_x(
        model,
        data_loader,
        each_x=False,
        val=curvature_type,
        output=["eigval"],
        apprx=approx_type,
    )
    eigvals = np.asarray(stats["eigval"], dtype=np.float64)
    return np.sort(eigvals)


def plot_eigen_spectrum(sorted_vals, path_out, mode_label):
    idx = np.arange(1, len(sorted_vals) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(idx, sorted_vals, label="After Training")
    plt.xlabel("Sorted index")
    plt.ylabel("Eigenvalue")
    plt.title(f"{mode_label} - eigenvalues after training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_out, dpi=150)
    plt.close()


def train(args):
    fisher_model = load_model_module()

    model = fisher_model.model
    optimizer = fisher_model.optimizer
    criterion = fisher_model.criterion

    dataset = TensorDataset(fisher_model.Xtr_t, fisher_model.ytr_t)
    loader = DataLoader(dataset, batch_size=fisher_model.batch_size, shuffle=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    eig_vals = None
    laplace_module = None
    laplace_loader = None
    if args.eig:
        laplace_module = load_laplace_module()
        laplace_loader = DataLoader(
            dataset, batch_size=fisher_model.batch_size, shuffle=False
        )

    train_accs, test_accs = [], []
    steps = []
    total_steps = 0
    batches_per_epoch = len(loader)
    batch_size = fisher_model.batch_size
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_acc = accuracy(model, fisher_model.Xtr_t, fisher_model.ytr_t)
        test_acc = accuracy(model, fisher_model.Xte_t, fisher_model.yte_t)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        total_steps += batches_per_epoch
        scaled_steps = total_steps * batch_size / args.epochs
        steps.append(scaled_steps)
        print(f"Epoch {epoch:03d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    if args.eig and laplace_module and laplace_loader:
        eig_vals = compute_eigenvalues(
            model, laplace_loader, laplace_module, args.val, args.apprx
        )
        eig_path = SAVE_DIR / "eigenval.json"
        with eig_path.open("w", encoding="utf-8") as f:
            json.dump(eig_vals.tolist(), f, indent=2)
        print(f"Saved eigenvalues after training to {eig_path}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    n_train = len(dataset)
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_accs, label="Train")
    plt.plot(steps, test_accs, label="Test")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    n_params = sum(p.numel() for p in model.parameters())
    final_test = test_accs[-1] if test_accs else 0.0
    plt.title(f"# data {n_train}, # params {n_params}, acc {final_test:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = PLOT_DIR / "acc.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    model_path = SAVE_DIR / "model_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model state dict to {model_path}")

    print(f"Saved accuracy plot to {plot_path}")
    if args.eig and eig_vals is not None:
        eig_filename = f"{args.apprx.lower()}_eigvals.png"
        eig_plot_path = PLOT_DIR / eig_filename
        plot_eigen_spectrum(
            eig_vals,
            eig_plot_path,
            mode_label=f"{args.apprx.lower()}",
        )
        print(f"Saved eigenvalue plot to {eig_plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Overparameterized NN toy experiment")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--val",
        type=str,
        default="GGN",
        choices=["EF", "GGN"],
        help="FIM variant to use when computing eigenvalues",
    )
    parser.add_argument(
        "--apprx",
        type=str,
        default="diag",
        choices=["diag", "full"],
        help="Laplace curvature approximation",
    )
    parser.add_argument(
        "--eig",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set to 1 to compute Laplace eigenvalues, 0 to skip",
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(0)
    args = parse_args()
    train(args)
