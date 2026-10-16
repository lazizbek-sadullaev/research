import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

import pysindy as ps

# --------------------------------------------------------
# 1. Biz sinamoqchi bugan sistema:
#    dx/dt = -0.1x + 2y
#    dy/dt = -2x - 0.1y       (not decoupled — birbiridan nomustaqil)
#
#    Yechim: x(t) = x0 * e^(-0.1t)
#            y(t) = y0 * e^(-0.1t) # menimcha tekshirish kerak bu yerda
# --------------------------------------------------------
# 1. True System to get X_dot
def linear_damped_oscillator(t, state): # (state, t) for odeint → (t, state) for solve_ivp
    x, y = state
    dxdt = -0.1 * x + 2.0 * y
    dydt = -2.0 * x - 0.1 * y
    return [dxdt, dydt]

# --- 2. Generate Data using 1. ---
# Vaqt massivi — uniform, 1-D array
# dt = 0.01 # uninform dt's
t_span = (0, 10)  # time span for integration [a, b]
# dt = 0.1 # nonuniform dt works or NOT?
# t  = np.arange(t_span[0], t_span[1], dt)   # shape: (1000,)  ← faqat 1-o'q: vaqt
# t_train = np.arange(0, 5, dt)

# --------------------------------------------------------
# Nonuniform (notekis oraliqli) vaqt massivi
# --------------------------------------------------------
rng = np.random.default_rng(42)
# np.random.default_rng(seed) — tasodifiy son generatori yaratadi.
# seed=42 — har safar bir xil "tasodifiy" natija chiqishi uchun
# (reproducibility — natijani qayta tekshirish oson bo'lishi uchun).

n_points = 100
t = rng.uniform(t_span[0], t_span[1], n_points)
# rng.uniform(past, yuqori, nechta) — belgilangan oraliqda
# n_points ta tasodifiy son tanlaydi, tartibsiz holda.

t = np.sort(t)
# np.sort() — massivni o'sish tartibida saralaydi.
# MUHIM: solve_ivp va model.fit() vaqt o'sib borishini talab qiladi,
# tasodifiy sonlar tartibsiz chiqadi, shuning uchun saralash SHART.

t[0] = t_span[0]
# Boshlang'ich nuqtani aniq t_span boshiga tenglashtiramiz
# (tasodifiy tanlov aniq 0 dan boshlanmasligi mumkin).
# print(f"nonuniform t = {t}")

# Boshlang'ich shartlar
x0 = [2.0, 0.0]  # Initial condition (approx from paper figure)

# Integrate to get state X = [x, y], usually this is gained in experiments or simuls
# X = odeint(linear_damped_oscillator, x0, t) # t_train replaced with t as opposed to DQE file
sol = solve_ivp(linear_damped_oscillator, t_span, x0, 
              t_eval=t, method="LSODA", 
              rtol=1e-12, atol=1e-12) # odeint replaced with solve_ivp as opposed to DQE file
if not sol.success:
    raise RuntimeError(sol.message)

X = sol.y.T     # (N, 2) ← .y shape (2, N), .T → (N, 2)
                # bizning np.stack((x,y), axis=-1) ga teng
# print("X=", X)
# Integrate each IC
# def integrate_lorenz(ic):
#     sol = solve_ivp(lorenz_system, t_span, ic,
#                     t_eval=t, method="LSODA",   # LSODA more stable for chaotic systems
#                     rtol=1e-12, atol=1e-12)
#     if not sol.success:
#         raise RuntimeError(sol.message)
#     return sol.y.T      # (N, 3)

# sols = [integrate_lorenz(ic) for ic in X0s]
# for i, s in enumerate(sols):
#     print(f"  IC {i+1:02d}: shape {s.shape}")

# X_all = np.vstack(sols)               # (n*N, 3)  ← stacked for OUR SINDy

print(f"Data is shape: {X.shape}")  # (1000, 2)
#                                       ^    ^
#                                     time  coord (x,y)
print(f"time is shape: {t.shape}")

# finite differences when noiseless data,
#  otherwise use smoothed finite differences or total variation regularization   
differentiation_method = ps.FiniteDifference(order=2)

# candidate library can be specified with an object from the feature_library submodule.
feature_library = ps.PolynomialLibrary(degree=3)

# Sequentially-thresholded least squares is the default choice of optimizer
optimizer = ps.STLSQ(threshold=0.01)

# Finally, we bring these three components together in one SINDy object.    
model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
)

# Following scikit-learn workflow, we first instantiate a SINDy class object 
# with the desired properties, then fit it to the data in separate step.
model.fit(X, t=t, feature_names=["x", "y"])

# Inspect the governing equations discovered by the model with the print method. 
# The output is a human-readable representation of the equations to check validity.
model.print()

# Alternate ways of passing data to fit() with a uniform time step:
# dt = t[1] - t[0]
# model.fit(X, t=dt, feature_names=["x", "y"])
# model.print()
# --------------------------------------------------------
# 10. Recovered (SINDy) sistemani simulyatsiya qilish
# --------------------------------------------------------
# model.simulate(x0, t) — bizning qo'lda yozgan sindy_stlsq_rhs()+odeint()
# juftligiga teng, lekin PySINDy buni ichkarida o'zi bajaradi:
# u model.fit() topgan Xi koeffitsientlaridan tenglama tuzib, integratsiya qiladi.
X_sim = model.simulate(x0, t)   # natija: (N, 2) — X bilan bir xil shakl

# --------------------------------------------------------
# 11. True vs Recovered — grafik solishtirish
# --------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# plt.subplots(1,2,...) — 1 qator, 2 ustunli grafik "grid" yaratadi.
# fig — butun rasm; axs — har biri alohida panel bo'lgan massiv (axs[0], axs[1]).

for i, var in enumerate(["x", "y"]):
    axs[i].plot(t, X[:, i], label="True", linewidth=2)
    axs[i].plot(t, X_sim[:, i], '--', label="SINDy (recovered)", linewidth=2)
    axs[i].set_xlabel("t")
    axs[i].set_ylabel(var)
    axs[i].set_title(f"{var}(t): True vs Recovered")
    axs[i].legend()

plt.tight_layout()
plt.show()

# Passing experimentally measured or true/known derivatives to model.fit()
# X — avval solve_ivp'dan olingan holat massivi, shakli (N, 2)
X_dot_true = np.array([linear_damped_oscillator(ti, xi) for ti, xi in zip(t, X)])
# shakli: (N, 2) — X bilan bir xil
model = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=3),
    optimizer=ps.STLSQ(threshold=0.01),
)
model.fit(X, t=t, x_dot=X_dot_true, feature_names=["x", "y"])
model.print()

# Multiple Trajectories
