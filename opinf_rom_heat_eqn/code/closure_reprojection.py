# closure_reprojection.py
# Experiment 2: what happens when the data do NOT lie in the POD subspace?
# Same heat equation, but the initial condition has a third mode (j = 5).
# We keep r = 2, so mode 5 is "outside" the reduced space.
# Run:  python closure_reprojection.py

import numpy as np
np.set_printoptions(precision=4, suppress=True)

N, dx, dt, K, r = 7, 1/8, 0.004, 40, 2
L = (np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)) / dx**2
A = np.eye(N) + dt*L
i = np.arange(1, N+1)

def fom_step(x):                    # the "black box": one time step of the FOM
    return A @ x

x0 = np.sin(np.pi*i/8) + 0.5*np.sin(3*np.pi*i/8) + 0.4*np.sin(5*np.pi*i/8)
X = np.zeros((N, K+1)); X[:, 0] = x0
for k in range(K):
    X[:, k+1] = fom_step(X[:, k])

U, s, _ = np.linalg.svd(X, full_matrices=False)
print("singular values:", s)
V = U[:, :r]

# (a) plain OpInf on projected FOM trajectories
Xh = V.T @ X
A_plain = np.linalg.lstsq(Xh[:, :-1].T, Xh[:, 1:].T, rcond=None)[0].T

# (b) OpInf with re-projection: lift, take ONE black-box step, project again
Xb = np.zeros((r, K+1)); Xb[:, 0] = V.T @ x0
for k in range(K):
    Xb[:, k+1] = V.T @ fom_step(V @ Xb[:, k])
A_reproj = np.linalg.lstsq(Xb[:, :-1].T, Xb[:, 1:].T, rcond=None)[0].T

# (c) intrusive Galerkin operator (needs A)
A_intr = V.T @ A @ V

print("plain OpInf:\n", A_plain)
print("re-projection OpInf:\n", A_reproj)
print("intrusive V^T A V:\n", A_intr)
print("||A_plain  - A_intr||_F = %.2e" % np.linalg.norm(A_plain - A_intr))
print("||A_reproj - A_intr||_F = %.2e" % np.linalg.norm(A_reproj - A_intr))
print("closure part of the data: max_k ||x_k - V V^T x_k|| = %.3e"
      % np.max(np.linalg.norm(X - V @ Xh, axis=0)))

# (d) prediction: run each 2x2 ROM from V^T x0, lift, compare with the FOM
def max_rel_error(M):
    xh = V.T @ x0
    errs = []
    for k in range(K+1):
        errs.append(np.linalg.norm(X[:, k] - V @ xh) / np.linalg.norm(X[:, k]))
        xh = M @ xh
    return max(errs)

print("max relative prediction error, plain OpInf  : %.4f" % max_rel_error(A_plain))
print("max relative prediction error, re-projection: %.4f" % max_rel_error(A_reproj))
print("max relative prediction error, intrusive    : %.4f" % max_rel_error(A_intr))
