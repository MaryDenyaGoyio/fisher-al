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