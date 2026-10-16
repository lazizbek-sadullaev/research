# --- 4.1.1. Define the System (Linear Damped Harmonic Oscillator) ---
# Method: STLS, sequential thresholded least square regression;
# From the paper: dx/dt = -0.1x + 2y, dy/dt = -2x - 0.1y

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# 1. True System to get X_dot
def linear_damped_oscillator(state, t):
    x, y = state
    dxdt = -0.1 * x + 2.0 * y
    dydt = -2.0 * x - 0.1 * y
    return [dxdt, dydt]

# --- 2. Generate Data using 1. ---
# dt = 0.01 # uninform dt's
dt = 1 # nonuniform dt works or NOT?
t_train = np.arange(0, 25, dt)
# t_train = np.arange(0, 5, dt)
x0 = [2.0, 0.0]  # Initial condition (approx from paper figure)

# Integrate to get state X = [x, y], usually this is gained in experiments or simuls
X = odeint(linear_damped_oscillator, x0, t_train)
# print("X=", X)

# Compute derivatives dX/dt (for SINDy targets), usually this is gained in experiments or simuls
# In practice, you might compute this numerically from X.
# Here we use the true function for clean demonstration.
X_dot = np.array([linear_damped_oscillator(s, ti) for s, ti in zip(X, t_train)])
# print("X_dot=", X_dot)

# --- 3. Construct Candidate Library (Theta) ---
# We include polynomials up to degree 3: [1, x, y, x^2, xy, y^2, x^3, ..., y^3]
x_val = X[:, 0]
y_val = X[:, 1]
print("x_val=", x_val)
print("y_val=", y_val)

# Building the library matrix manually for clarity
Theta = np.column_stack([
    np.ones_like(x_val),          # 1
    x_val,                        # x
    y_val,                        # y
    x_val**2,                     # x^2
    x_val * y_val,                # xy
    y_val**2,                     # y^2
    x_val**3,                     # x^3
    (x_val**2) * y_val,           # x^2y
    x_val * (y_val**2),           # xy^2
    y_val**3                      # y^3
])

feature_names = ['1', 'x', 'y', 'x^2', 'xy', 'y^2', 'x^3', 'x^2y', 'xy^2', 'y^3']
print("Theta.shape=", Theta.shape)

# --- 4. Sparse Regression (STLS Algorithm) ---
# Solve X_dot = Theta * Xi with sparsity constraint, sequential threshold least square regression
def sindy_stls(Theta, X_dot, threshold=0.05, max_iter=10):
    # Initial Least Squares guess
    Xi = np.linalg.lstsq(Theta, X_dot, rcond=None)[0]

    for k in range(max_iter):
        # Identify small coefficients (below threshold), indeed need to use optimzn to minimize the LASSO operator
        small_inds = np.abs(Xi) < threshold
        Xi[small_inds] = 0  # Enforce sparsity, instead of LASSO

        # Regress again only on the remaining non-zero terms for each column
        for ind in range(X_dot.shape[1]): # For x and y equations separately
            big_inds = ~small_inds[:, ind]
            if np.sum(big_inds) > 0:
                Xi[big_inds, ind] = np.linalg.lstsq(Theta[:, big_inds], X_dot[:, ind], rcond=None)[0]
    return Xi

# Run SINDy
Xi = sindy_stls(Theta, X_dot, threshold=0.05)

# --- 5. Display Results ---
df_results = pd.DataFrame(Xi, index=feature_names, columns=['dx/dt', 'dy/dt'])

print("Original System Matrix (from paper):")
print("[[-0.1,  2.0],")
print(" [-2.0, -0.1]]")
print("\nSINDy Identified Coefficients (Sparse Model):")
# Filter out zero rows for cleaner view if desired, or show all
display(df_results)

# --- 6. Plotting ---
plt.figure(figsize=(12, 5))

# Time Series
plt.subplot(1, 2, 1)
plt.plot(t_train, X[:, 0], label='x')
plt.plot(t_train, X[:, 1], label='y')
plt.title("Time Series Data (Generated)")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend()

# Phase Portrait
plt.subplot(1, 2, 2)
plt.plot(X[:, 0], X[:, 1], 'k')
plt.title("Phase Portrait (x vs y)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

plt.tight_layout()
plt.show()