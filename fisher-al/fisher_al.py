import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from .laplace_redux import I_theta_x


def optimality(
    model,
    data_acq,
    val: str = "GGN",
    output=("A_opt", "D_opt"),
    apprx: str = "diag",
    rank: int = 100,
    precision: float = 1e-4,
):
    """
    Compute A- and D-optimality statistics from the Fisher information of data_acq.
    """

    stats = I_theta_x(
        model,
        data_acq,
        each_x=False,
        val=val,
        output=["eigval"],
        apprx=apprx,
        rank=rank,
    )

    eigvals = np.asarray(stats["eigval"], dtype=np.float64)
    eigvals = np.maximum(eigvals, precision)

    results = {}
    if "A_opt" in output:
        results["A_opt"] = float(1.0 / np.sum(1.0 / eigvals))
    if "D_opt" in output:
        results["D_opt"] = float(np.sum(np.log(eigvals)))
    if "A_inv_opt" in output:
        results["A_inv_opt"] = float(np.sum(eigvals))
    return results


def _ensure_loader(data, batch_size=None, shuffle=False):
    if isinstance(data, DataLoader):
        return data
    if hasattr(data, "__len__") and hasattr(data, "__getitem__"):
        return DataLoader(data, batch_size=batch_size or len(data), shuffle=shuffle)
    raise TypeError("data must be a DataLoader or Dataset.")


def _predict_with_model(model, data_pool, batch_size=256):
    loader = _ensure_loader(data_pool, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    xs, ys = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            logits = model(xb)
            y_pred = logits.argmax(dim=1).cpu()
            xs.append(xb.cpu())
            ys.append(y_pred)

    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return TensorDataset(X, y)


def A_acq(
    model,
    data_acq,
    data_pool,
    val: str = "GGN",
    apprx: str = "diag",
    rank: int = 100,
    precision: float = 1e-4,
):
    """
    Compute A-optimality acquisition scores for each (x, y_pred) in data_pool.
    """

    data_acq_loader = _ensure_loader(data_acq, shuffle=False)
    pool_with_preds = _predict_with_model(model, data_pool)
    combined = ConcatDataset([data_acq_loader.dataset, pool_with_preds])
    combined_loader = DataLoader(
        combined, batch_size=len(combined), shuffle=False
    )

    stats_per_sample = I_theta_x(
        model,
        combined_loader,
        each_x=True,
        val=val,
        output=["eigval"],
        apprx=apprx,
        rank=rank,
    )

    n_pool = len(pool_with_preds)
    acq_values = []
    for stats in stats_per_sample[-n_pool:]:
        eigvals = np.asarray(stats["eigval"], dtype=np.float64)
        eigvals = np.maximum(eigvals, precision)
        acq_values.append(float(1.0 / np.sum(1.0 / eigvals)))

    return {"A_acq": acq_values}


def A_inv_acq(
    model,
    data_acq,
    data_pool,
    val: str = "GGN",
    apprx: str = "diag",
    rank: int = 100,
    precision: float = 1e-4,
):
    """
    Compute trace(I) acquisition scores for each (x, y_pred) in data_pool.
    """

    data_acq_loader = _ensure_loader(data_acq, shuffle=False)
    pool_with_preds = _predict_with_model(model, data_pool)
    combined = ConcatDataset([data_acq_loader.dataset, pool_with_preds])
    combined_loader = DataLoader(
        combined, batch_size=len(combined), shuffle=False
    )

    stats_per_sample = I_theta_x(
        model,
        combined_loader,
        each_x=True,
        val=val,
        output=["eigval"],
        apprx=apprx,
        rank=rank,
    )

    n_pool = len(pool_with_preds)
    acq_values = []
    for stats in stats_per_sample[-n_pool:]:
        eigvals = np.asarray(stats["eigval"], dtype=np.float64)
        eigvals = np.maximum(eigvals, precision)
        acq_values.append(float(np.sum(eigvals)))

    return {"A_inv_acq": acq_values}


def D_acq(
    model,
    data_acq,
    data_pool,
    val: str = "GGN",
    apprx: str = "diag",
    rank: int = 100,
    precision: float = 1e-4,
):
    """
    Compute D-optimality acquisition scores (log det) for each (x, y_pred) in data_pool.
    """

    data_acq_loader = _ensure_loader(data_acq, shuffle=False)
    pool_with_preds = _predict_with_model(model, data_pool)
    combined = ConcatDataset([data_acq_loader.dataset, pool_with_preds])
    combined_loader = DataLoader(
        combined, batch_size=len(combined), shuffle=False
    )

    stats_per_sample = I_theta_x(
        model,
        combined_loader,
        each_x=True,
        val=val,
        output=["eigval"],
        apprx=apprx,
        rank=rank,
    )

    n_pool = len(pool_with_preds)
    acq_values = []
    for stats in stats_per_sample[-n_pool:]:
        eigvals = np.asarray(stats["eigval"], dtype=np.float64)
        eigvals = np.maximum(eigvals, precision)
        acq_values.append(float(np.sum(np.log(eigvals))))

    return {"D_acq": acq_values}
