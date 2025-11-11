
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, Subset
from laplace import Laplace
from utils.LaplaceUtils import return_hessian_eigenvalues, compute_outcome_hessian_from_model, symmetric_matrix_sqrt, fast_jacobian, low_rank_updated_part

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
        A_inv = torch.linalg.inv(A)            # (d_out, d_out)
        V = C @ symmetric_matrix_sqrt(A_inv)                    
        deltas.append(torch.sum(V**2))
    return torch.stack(deltas)  # (b,)

def ToptScore_per_sample(model, x, Hessian, eps=1e-10):
    """
    Compute T-opt reduction per input sample. (Just Trace reduction)
    Returns: torch tensor shape (b,) with per-sample Delta values.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    U_batch = low_rank_updated_part(model, x, return_batch=True)   # (b, n_params, d_out)

    deltas = []
    for i in range(U_batch.shape[0]):
        U_i = U_batch[i]                    # (n_params, d_out)
        C = U_i                              # (n_params, d_out)
        delta = torch.sum(C**2, dim=(0,1))  # scalar
        deltas.append(delta)

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
    elif score_type == 'Topt':
        scores = ToptScore_per_sample(model, unlabeled_set[:][0], Hessian)
    else:
        raise ValueError("score_type Does Not matches")

    # Get top_k indices
    topk_indices = torch.topk(scores, k=top_k).indices
    return topk_indices.tolist()

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