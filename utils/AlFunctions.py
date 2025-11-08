
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, Subset
from laplace import Laplace

def fast_jacobian(model, x):
    model.zero_grad()
    # ensure batch dimension
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1, input_dim)
    b = x.shape[0]

    params = [p for p in model.parameters() if p.requires_grad]
    # precompute param sizes to flatten consistently
    param_numels = [p.numel() for p in params]

    J_batch = []
    for i in range(b):
        xi = x[i : i + 1]  # keep batch dim for forward
        yi = model(xi)  # (1, d_out)
        d_out = yi.shape[-1]

        # compute grads for this sample: result (d_out, n_params)
        J_i_rows = []
        for k in range(d_out):
            grads = torch.autograd.grad(yi[0, k], params, retain_graph=True)
            grad_flat = torch.cat([g.reshape(-1) for g in grads])
            J_i_rows.append(grad_flat)
        J_i = torch.stack(J_i_rows, dim=0)  # (d_out, n_params)
        J_batch.append(J_i)

    J_batch = torch.stack(J_batch, dim=0)  # (b, d_out, n_params)
    return J_batch

def compute_outcome_hessian_from_model(model, inputs):
    # inputs: (d,) or (b, d)
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    z = model(inputs)  # (b, d_out)
    p = torch.softmax(z, dim=-1)  # (b, d_out)
    # diag_embed builds (b, d_out, d_out)
    H = torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)  # (b, d_out, d_out)
    return H

def symmetric_matrix_sqrt(A, eps=1e-12):
    """
    A: (n,n) or (batch, n, n)
    returns: A_sqrt with same shape
    """
    single = (A.dim() == 2)
    if single:
        A = A.unsqueeze(0)
    w, v = torch.linalg.eigh(A)
    w_clamped = torch.clamp(w, min=eps)
    w_sqrt = torch.sqrt(w_clamped)
    A_sqrt = (v * w_sqrt.unsqueeze(-2)) @ v.transpose(-2, -1)
    if single:
        return A_sqrt[0]
    return A_sqrt

def low_rank_updated_part(model, x, return_batch: bool = False):
    """
    Returns:
      - if return_batch=True: U_batch of shape (b, n_params, d_out)
      - else: U_all of shape (n_params, b * d_out)  (backward-compatible)
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    H = compute_outcome_hessian_from_model(model, x)    # (b, d_out, d_out)
    J_batch = fast_jacobian(model, x)                   # (b, d_out, n_params)
    H_sqrt = symmetric_matrix_sqrt(H)                   # (b, d_out, d_out)

    # J_batch: (b, d_out, n_params) -> transpose -> (b, n_params, d_out)
    Jt = J_batch.transpose(1, 2)
    U_batch = torch.matmul(Jt, H_sqrt)                  # (b, n_params, d_out)

    if return_batch:
        return U_batch
    b, n_params, d_out = U_batch.shape
    U_all = U_batch.permute(1, 0, 2).reshape(n_params, b * d_out)
    return U_all

def DoptScore_per_sample(model, x, Hessian, eps=1e-10):
    """
    Compute D-opt score per input sample.
    - x: (input_dim,) or (b, input_dim)
    - Hessian: torch tensor (n_params,) (diagonal)
    Returns: torch tensor shape (b,) with per-sample log-determinant scores
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    U_batch = low_rank_updated_part(model, x, return_batch=True)   # (b, n_params, d_out)
    Hinv = 1.0 / (Hessian + eps)                                   # (n_params,)

    scores = []
    for i in range(U_batch.shape[0]):
        U_i = U_batch[i]                    # (n_params, d_out)
        C = Hinv.unsqueeze(1) * U_i         # (n_params, d_out)
        A = torch.eye(U_i.shape[1], device=U_i.device) + (U_i.T @ C)  # (d_out, d_out)
        # use slogdet for stability
        sign, ld = torch.linalg.slogdet(A)
        # if numeric issue (sign <=0) return nan for that sample
        scores.append(ld if sign > 0 else torch.tensor(float('nan'), device=A.device))
    return torch.stack(scores)  # (b,)

def AoptScore_per_sample(model, x, Hessian, eps=1e-10):
    """
    Compute A-opt reduction per input sample.
    Returns: torch tensor shape (b,) with per-sample Delta values.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    U_batch = low_rank_updated_part(model, x, return_batch=True)   # (b, n_params, d_out)
    Hinv = 1.0 / (Hessian + eps)                                   # (n_params,)

    deltas = []
    for i in range(U_batch.shape[0]):
        U_i = U_batch[i]                    # (n_params, d_out)
        C = Hinv.unsqueeze(1) * U_i         # (n_params, d_out)
        A = torch.eye(U_i.shape[1], device=U_i.device) + (U_i.T @ C)  # (d_out, d_out)
        A = A + eps * torch.eye(A.shape[0], device=A.device)
        S = C.T @ C                          # (d_out, d_out)
        X = torch.linalg.solve(A, S)         # (d_out, d_out)
        deltas.append(torch.trace(X))
    return torch.stack(deltas)  # (b,)

def selection_AL_scores(model, unlabeled_set, Hessian, score_type='Aopt', top_k=10):
    """
    Select top_k samples from unlabeled_set based on AL scores.
    score_type: 'Aopt' or 'Dopt'
    Returns: indices of selected samples in unlabeled_set
    """
    if score_type == 'Aopt':
        scores = AoptScore_per_sample(model, unlabeled_set[:][0], Hessian)
    elif score_type == 'Dopt':
        scores = DoptScore_per_sample(model, unlabeled_set[:][0], Hessian)
    else:
        raise ValueError("score_type must be 'Aopt' or 'Dopt'")

    # Get top_k indices
    topk_indices = torch.topk(scores, k=top_k).indices
    return topk_indices.tolist()

def extract_samples_from_unlabeled(unlabeled_set, selected_indices):
    """
    Extract samples at selected_indices from unlabeled_set.
    Returns: list of (input, label) tuples.
    """
    samples = [unlabeled_set[i] for i in selected_indices]
    return samples

def delete_samples_from_unlabeled(unlabeled_set, selected_indices):
    """
    Remove samples at selected_indices from unlabeled_set.
    Returns: new Subset of unlabeled_set without selected samples.
    """
    all_indices = list(range(len(unlabeled_set)))
    remaining_indices = [i for j, i in enumerate(all_indices) if j not in selected_indices]
    new_unlabeled_set = Subset(unlabeled_set.dataset, [unlabeled_set.indices[i] for i in remaining_indices])
    return new_unlabeled_set

def add_samples_to_labeled(labeled_set, unlabeled_set, selected_indices):
    """
    Add samples at selected_indices from unlabeled_set to labeled_set.
    Returns: new Subset of labeled_set with added samples.
    """
    new_indices = labeled_set.indices + [unlabeled_set.indices[i] for i in selected_indices]
    new_labeled_set = Subset(labeled_set.dataset, new_indices)
    return new_labeled_set

def add_and_extract_and_delete_samples(labeled_set, unlabeled_set, selected_indices):
    new_labeled_set = add_samples_to_labeled(labeled_set, unlabeled_set, selected_indices)
    extracted_samples = extract_samples_from_unlabeled(unlabeled_set, selected_indices)
    new_unlabeled_set = delete_samples_from_unlabeled(unlabeled_set, selected_indices)
    return new_labeled_set, new_unlabeled_set, extracted_samples

def AL_finetune_model(model, AL_train_set_Labeled, criterion, optimizer, n_epochs=10):
    for epoch in range(n_epochs):
        model.train()
        for inputs, labels in AL_train_set_Labeled:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
