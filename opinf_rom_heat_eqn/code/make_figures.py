# make_figures.py
# Makes the five figures of the note and saves them in ../note/figures/.
# Run:  python make_figures.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "note", "figures")
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

N, dx, dt, K, r = 7, 1/8, 0.004, 40, 2
L = (np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)) / dx**2
A = np.eye(N) + dt*L
i = np.arange(1, N+1)
xgrid = np.linspace(0, 1, N+2)               # grid with the two boundary points

def run(x0):
    X = np.zeros((N, K+1)); X[:, 0] = x0
    for k in range(K):
        X[:, k+1] = A @ X[:, k]
    return X

def with_bc(x):                              # add the zero boundary values
    return np.concatenate(([0.0], x, [0.0]))

x0 = np.sin(np.pi*i/8) + 0.5*np.sin(3*np.pi*i/8)
X = run(x0)
U, s, _ = np.linalg.svd(X, full_matrices=False)
V = U[:, :r]
Xh = V.T @ X
A_op = np.linalg.lstsq(Xh[:, :-1].T, Xh[:, 1:].T, rcond=None)[0].T

# Figure 1: snapshots of the FOM
fig, ax = plt.subplots(figsize=(5.5, 3.2))
for k, c in zip([0, 5, 10, 20, 40], plt.cm.viridis(np.linspace(0, 0.9, 5))):
    ax.plot(xgrid, with_bc(X[:, k]), "o-", color=c, ms=4, label=f"k = {k}")
ax.set_xlabel("position  $\\xi$"); ax.set_ylabel("temperature  $x_k$")
ax.set_title("Full-order model: 7 interior points, explicit Euler")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_snapshots.pdf")); plt.close(fig)

# Figure 2: singular values (example 1 and example 2)
x0b = x0 + 0.4*np.sin(5*np.pi*i/8)
sb = np.linalg.svd(run(x0b), compute_uv=False)
fig, ax = plt.subplots(figsize=(5.5, 3.2))
eps = 1e-17
ax.semilogy(np.arange(1, N+1), np.maximum(s, eps), "o-", label="Example 1: modes 1, 3")
ax.semilogy(np.arange(1, N+1), np.maximum(sb, eps), "s--", label="Example 2: modes 1, 3, 5")
ax.set_xlabel("index $j$"); ax.set_ylabel("singular value $\\sigma_j$")
ax.set_title("Singular values of the snapshot matrix")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_singular_values.pdf")); plt.close(fig)

# Figure 3: POD modes versus the normalized sine vectors q1, q3
q1, q3 = np.sin(np.pi*i/8)/2, np.sin(3*np.pi*i/8)/2
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(xgrid, with_bc(V[:, 0]*np.sign(V[3, 0])), "o-", label="POD mode 1")
ax.plot(xgrid, with_bc(V[:, 1]*np.sign(V[0, 1])), "s-", label="POD mode 2")
ax.plot(xgrid, with_bc(q1), "k:", label="$q_1$ (sine, $j=1$)")
ax.plot(xgrid, with_bc(q3), "k--", label="$q_3$ (sine, $j=3$)")
ax.set_xlabel("position  $\\xi$"); ax.set_title("POD modes are slightly rotated sines")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_pod_modes.pdf")); plt.close(fig)

# Figure 4: reduced coordinates, data vs OpInf prediction
xh = V.T @ x0; P = [xh]
for k in range(K):
    xh = A_op @ xh; P.append(xh)
P = np.array(P).T
fig, ax = plt.subplots(figsize=(5.5, 3.2))
kk = np.arange(K+1)
ax.plot(kk, Xh[0], "o", ms=4, mfc="none", label="data $\\hat{x}_{k,1}$")
ax.plot(kk, P[0], "-", label="OpInf ROM, coord. 1")
ax.plot(kk, Xh[1], "s", ms=4, mfc="none", label="data $\\hat{x}_{k,2}$")
ax.plot(kk, P[1], "-", label="OpInf ROM, coord. 2")
ax.set_xlabel("time step $k$"); ax.set_ylabel("reduced coordinate")
ax.set_title("Example 1: the 2x2 ROM reproduces the data")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_reduced_coords.pdf")); plt.close(fig)

# Figure 5: Example 2, relative error of the three ROMs over time
Xb = run(x0b)
Ub = np.linalg.svd(Xb, full_matrices=False)[0]; Vb = Ub[:, :r]
Xhb = Vb.T @ Xb
A_plain = np.linalg.lstsq(Xhb[:, :-1].T, Xhb[:, 1:].T, rcond=None)[0].T
A_intr = Vb.T @ A @ Vb
proj_err = np.linalg.norm(Xb - Vb @ Xhb, axis=0) / np.linalg.norm(Xb, axis=0)
def errs(M):
    xh = Vb.T @ x0b; e = []
    for k in range(K+1):
        e.append(np.linalg.norm(Xb[:, k] - Vb @ xh) / np.linalg.norm(Xb[:, k])); xh = M @ xh
    return np.array(e)
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.semilogy(kk, errs(A_plain), "-", lw=2, label="plain OpInf ROM")
ax.semilogy(kk, errs(A_intr), "--", label="intrusive ROM")
ax.semilogy(kk, np.maximum(proj_err, 1e-17), "k:", label="best possible: $\\|x_k - VV^Tx_k\\|/\\|x_k\\|$")
ax.set_xlabel("time step $k$"); ax.set_ylabel("relative error")
ax.set_title("Example 2 (r = 2): the error comes from truncation")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_example2_errors.pdf")); plt.close(fig)

print("figures written to", os.path.abspath(out))
