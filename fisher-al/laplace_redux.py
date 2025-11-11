import numpy as np
import torch
import torch
from laplace import Laplace


def I_theta_x(model, data, each_x: bool = False, val: str = 'GGN', output=['eigval'], apprx: str = 'diag', rank = 100, precision=0.1):
    """
    EF:     F(θ; D) = ∑^n_i=1 ∇_θ log p(y_i|x_i,θ) ∇_θ log p(y_i|x_i,θ)^T
    GGN:    G(θ; D) = ∑^n_i=1 ∇_θ f(x_i) ∇²_z log p(y_i|z_i) ∇_θ f(x_i)^T

    Args:
        model: NN
        data: (x, y)^n
        each_x: bool
        val: 'EF' or 'GGN'
        output: ['eigval', 'eigvector', 'trace', 'logdet']
        apprx: 'diag', 'kfac', 'lowrank', 'full'

    Returns:
        {'eigval': array, 'trace': float, 'logdet': float, 'eigvector': array}
    """

    # ------------------------ check args ------------------------
    if invalid := set(output) - {'eigval', 'eigvector', 'trace', 'logdet'}: raise ValueError(f"[Error] Unsupported output keys: {invalid}")
    if (apprx := apprx.lower()) not in {'diag', 'kfac', 'lowrank', 'full'}: raise ValueError(f"[Error] Unsupported approximation '{apprx}'.")
    if apprx == 'lowrank' and (val := val.upper()) == 'EF': raise ValueError("[Error] Low-rank approximation only available with GGN curvature.")
    if each_x and not hasattr(data, 'dataset'): raise ValueError("[Error] Data loader must have a dataset attribute for per-sample mode.")

    # ------------------------ fit Laplace ------------------------
    likelihood = 'regression' if torch.is_floating_point(next(iter(data))[1]) else 'classification'

    backend, backend_kwargs = None, None
    if apprx == 'lowrank':
        from laplace.curvature.asdfghjkl import AsdfghjklHessian
        backend = AsdfghjklHessian
        backend_kwargs = {'low_rank': rank}
    elif val == 'EF':
        from laplace.curvature.curvlinops import CurvlinopsEF
        backend = CurvlinopsEF

    model_mode = model.training
    la = Laplace(
        model,
        likelihood,
        subset_of_weights='all',
        hessian_structure=apprx,
        backend=backend,
        backend_kwargs=backend_kwargs,
        prior_precision=precision,
    )
    la.fit(data, override=True, progress_bar=False)
    model.train(model_mode)

    # ================================================================================
    #                                   compute FIM
    # ================================================================================

    # ------------------------ λ, v, tr, det ------------------------
    def eig(curvature) -> dict[str, object]:
        
        def _safe_logdet(arr: np.ndarray) -> float: return float(np.log(np.maximum(arr, 1e-12)).sum()) if arr.size else float('-inf')

        # ------------------------ 1) diag ------------------------
        if apprx == 'diag':
            eigvals_np = curvature.detach().cpu().numpy()
            if 'eigvector' in output:   eigvecs_out = np.eye(len(eigvals_np), dtype=eigvals_np.dtype)

        # ------------------------ 2) kfac ------------------------
        elif apprx == 'kfac':
            from laplace.utils.matrix import Kron, KronDecomposed

            kron = curvature.decompose() if isinstance(curvature, Kron) else curvature
            if not isinstance(kron, KronDecomposed): raise TypeError("[Error] Unexpected KFAC structure.")
            pieces = []
            for factors, delta in zip(kron.eigenvalues, kron.deltas):
                if len(factors) == 1:   pieces.append((factors[0] + delta).reshape(-1))
                elif len(factors) == 2: pieces.append(torch.outer(factors[0], factors[1]).reshape(-1) + delta)
                else:                   raise ValueError("[Error] Unsupported KFAC config.")
            
            eigvals_t = torch.cat(pieces) if pieces else torch.empty(0)
            eigvals_np = eigvals_t.detach().cpu().numpy()

            if 'eigvector' in output:   eigvecs_out = [[vec.detach().cpu().numpy() for vec in group] for group in kron.eigenvectors]
            logdet_val = float(kron.logdet().item())

        # ------------------------ 3) low ------------------------
        elif apprx == 'lowrank':
            (vecs, eigvals_lr), _ = curvature
            eigvals_np = eigvals_lr.detach().cpu().numpy()
            if 'eigvector' in output:   eigvecs_out = vecs.detach().cpu().numpy()

        # ------------------------ 4) full ------------------------
        else:  # full
            eigvals_t, eigvecs_t = torch.linalg.eigh(curvature)
            eigvals_np = eigvals_t.detach().cpu().numpy()
            if 'eigvector' in output:   eigvecs_out = eigvecs_t.detach().cpu().numpy()

        stat = {}
        if 'eigval' in output: stat['eigval'] = eigvals_np
        if 'eigvector' in output: stat['eigvector'] = eigvecs_out
        if 'trace' in output: stat['trace'] = float(eigvals_np.sum())
        if 'logdet' in output: stat['logdet'] = _safe_logdet(eigvals_np) if not apprx == 'kfac' else logdet_val
        return stat

    # ------------------------ I(θ) ------------------------
    posterior = la.posterior_precision
    result = eig(posterior)

    if not each_x:
        return result

    # ------------------------ I(θ; x) ------------------------
    N = len(data.dataset)
    device = next(model.parameters()).device

    stats = []
    for xb, yb, *rest in data:
        xb = (xb if isinstance(xb, torch.Tensor) else torch.as_tensor(xb)).to(device)
        yb = (yb if isinstance(yb, torch.Tensor) else torch.as_tensor(yb)).to(device)
        for idx in range(len(xb)):
            x_i = xb[idx : idx + 1]
            y_i = yb[idx : idx + 1]

            if apprx == 'diag':
                _, curvature = la.backend.diag(x_i, y_i, N=N)
                stats.append(eig(curvature))
            elif apprx == 'kfac':
                _, curvature = la.backend.kron(x_i, y_i, N=N)
                stats.append(eig(curvature))
            else:  # full
                _, curvature = la.backend.full(x_i, y_i, N=N)
                stats.append(eig(curvature))

    return stats
