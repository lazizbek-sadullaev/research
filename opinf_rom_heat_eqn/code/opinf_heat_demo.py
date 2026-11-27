# opinf_heat_demo.py
# Operator Inference (OpInf) on a tiny heat equation: 7 grid points, r = 2.
# Run:  python opinf_heat_demo.py

import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ---- Block 1: the full-order model (FOM) --------------------------------
N, dx, dt = 7, 1/8, 0.004            # 7 interior points on [0,1], time step
L = (np.diag(-2*np.ones(N))          # main diagonal: -2
     + np.diag(np.ones(N-1), 1)      # upper diagonal: +1
     + np.diag(np.ones(N-1), -1)) / dx**2   # lower diagonal: +1
A = np.eye(N) + dt*L                 # explicit Euler:  x_{k+1} = A x_k

# ---- Block 2: initial condition and snapshots ---------------------------
i = np.arange(1, N+1)                # grid index 1,...,7  (x_i = i/8)
x0 = np.sin(np.pi*i/8) + 0.5*np.sin(3*np.pi*i/8)

K = 40                               # number of time steps
X = np.zeros((N, K+1))               # snapshot matrix, one column per step
X[:, 0] = x0
for k in range(K):
    X[:, k+1] = A @ X[:, k]

# ---- Block 3: POD basis from the SVD of the snapshots ------------------
U, s, Wt = np.linalg.svd(X, full_matrices=False)
print("singular values:", s)
r = 2
V = U[:, :r]                         # POD basis, N x r

# ---- Block 4: project the snapshots ------------------------------------
Xh = V.T @ X                         # reduced snapshots, r x (K+1)

# ---- Block 5: Operator Inference = least squares -----------------------
D = Xh[:, :-1].T                     # rows: xhat_k^T,     k = 0..K-1
R = Xh[:, 1:].T                      # rows: xhat_{k+1}^T, k = 0..K-1
AhatT, res, rank, sv = np.linalg.lstsq(D, R, rcond=None)
A_opinf = AhatT.T                    # we solved D Ahat^T = R, so transpose

# ---- Block 6: compare with the intrusive (Galerkin) reduced operator ----
A_intr = V.T @ A @ V                 # needs A  ->  intrusive
print("OpInf A_hat:\n", A_opinf)
print("Intrusive V^T A V:\n", A_intr)
print("||A_opinf - A_intr||_F =", np.linalg.norm(A_opinf - A_intr))
print("eigenvalues of A_opinf:", np.linalg.eigvals(A_opinf))

# ---- Block 7: predict with the ROM and lift back to the full space -----
xh = V.T @ x0
for k in range(K):
    xh = A_opinf @ xh
print("FOM  x_K   :", X[:, K])
print("ROM  V xh_K:", V @ xh)
print("max error  :", np.max(np.abs(X[:, K] - V @ xh)))
