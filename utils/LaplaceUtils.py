import laplace
import torch
from laplace import Laplace
from torch.utils.data import TensorDataset, DataLoader, Subset

def return_hessian_eigenvalues(model, train_set_Labeled):
    la = Laplace(model, 'classification', subset_of_weights='all', hessian_structure='diag')
    train_loader = DataLoader(train_set_Labeled, batch_size=32, shuffle=True)
    la.fit(train_loader)
    val = la.H
    return val

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