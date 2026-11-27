# hand_check.py
# Checks, one by one, every number of the hand calculation (Section 7 of the note).
# Run:  python hand_check.py

import numpy as np
np.set_printoptions(precision=4, suppress=True)

N, dx, dt, K = 7, 1/8, 0.004, 40
i = np.arange(1, N+1)
L = (np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)) / dx**2
A = np.eye(N) + dt*L

# Step 1: the key number dt/dx^2
print("Step 1: dt/dx^2 =", dt/dx**2)

# Step 2: eigenvalues a_j = 1 - 4 (dt/dx^2) sin^2(j pi/16)
a = {j: 1 - 4*dt/dx**2*np.sin(j*np.pi/16)**2 for j in (1, 3)}
print("Step 2: a1 = %.6f, a3 = %.6f" % (a[1], a[3]))
v1, v3 = np.sin(np.pi*i/8), np.sin(3*np.pi*i/8)
print("        A v1 / v1 =", (A @ v1) / v1)
print("        A v3 / v3 =", (A @ v3) / v3)

# Step 3: norms and orthogonality, q_j = v_j / 2
print("Step 3: ||v1||^2 = %.4f, ||v3||^2 = %.4f, v1.v3 = %.1e" % (v1 @ v1, v3 @ v3, v1 @ v3))
q1, q3 = v1/2, v3/2

# Step 4: closed-form trajectory x_k = 2 a1^k q1 + a3^k q3
X = np.zeros((N, K+1)); X[:, 0] = v1 + 0.5*v3
for k in range(K):
    X[:, k+1] = A @ X[:, k]
k = np.arange(K+1)
X_formula = np.outer(q1, 2*a[1]**k) + np.outer(q3, a[3]**k)
print("Step 4: max |X - formula| = %.1e" % np.max(np.abs(X - X_formula)))

# Step 5: Gram matrix G = C C^T with geometric sums, and singular values
G11 = 4*(1 - a[1]**(2*(K+1))) / (1 - a[1]**2)
G33 = (1 - a[3]**(2*(K+1))) / (1 - a[3]**2)
G13 = 2*(1 - (a[1]*a[3])**(K+1)) / (1 - a[1]*a[3])
print("Step 5: G11 = %.4f, G33 = %.4f, G13 = %.4f" % (G11, G33, G13))
m, d = (G11 + G33)/2, np.sqrt(((G11 - G33)/2)**2 + G13**2)
lam1, lam2 = m + d, m - d
print("        eigenvalues of G: %.4f, %.4f" % (lam1, lam2))
print("        sigma1 = %.4f, sigma2 = %.4f" % (np.sqrt(lam1), np.sqrt(lam2)))
print("        numpy svd(X):", np.linalg.svd(X, compute_uv=False)[:3])

# Step 6: rotation angle, tan(2 theta) = 2 G13 / (G11 - G33)
theta = 0.5*np.arctan2(2*G13, G11 - G33)
c, s = np.cos(theta), np.sin(theta)
print("Step 6: theta = %.4f rad = %.3f deg, cos = %.4f, sin = %.4f" % (theta, np.degrees(theta), c, s))

# Step 7: the reduced operator from the hand formulas
A11 = a[1]*c**2 + a[3]*s**2
A22 = a[1]*s**2 + a[3]*c**2
A12 = (a[1] - a[3])*c*s
print("Step 7: A11 = %.4f, A22 = %.4f, |A12| = %.4f" % (A11, A22, A12))
print("        trace = %.4f (a1+a3 = %.4f), det = %.4f (a1*a3 = %.4f)"
      % (A11 + A22, a[1] + a[3], A11*A22 - A12**2, a[1]*a[3]))
U = np.linalg.svd(X, full_matrices=False)[0]
V = U[:, :2]
print("        numpy V^T A V:\n", V.T @ A @ V)
