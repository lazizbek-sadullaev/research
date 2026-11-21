"""
Compressed Sensing (Donoho, 2006) -- kichik misollar / small examples
======================================================================
Bu fayl slaydlardagi BARCHA sonlarni va rasmlarni hosil qiladi.
This file produces ALL numbers and figures used on the slides.

Ishga tushirish / run:
    python cs_examples.py        (rasm yozuvlari o'zbekcha, *_uz.pdf)
    python cs_examples.py en     (figure labels in English, *_en.pdf)

Bo'limlar / sections:
  A) l_p "norma" va sparsity                  B) N-term approximation
  C) Haar transform (m = 4)                    D) Gel'fand width: romb misoli
  E) 2x3 matritsa: (P1) qachon topadi, qachon topmaydi
  F) Asosiy tajriba: m=256, n=64, 8 ta nolmas  G) muvaffaqiyat ehtimoli n ga qarab
  H) compressible signal: xato n ga qarab      I) CS1 va CS2 shartlari sonlarda
  J) shovqinga barqarorlik                     K) Lorenz library uchun CS1
  L) "stylized applications" sonlari
"""
import sys
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.integrate import odeint

np.set_printoptions(precision=3, suppress=True)
FIG = "../figures/"
LANG = sys.argv[1] if len(sys.argv) > 1 else "uz"
LABELS = {
    "uz": {"true": "haqiqiy x", "l1": "min L1 (P1): aniq tiklandi", "l2": "min L2: yoyilgan",
           "idx": "koordinata indeksi", "val": "qiymat",
           "n": "o'lchovlar soni n", "succ": "aniq tiklash ulushi",
           "k5": "5 ta nolmas", "k10": "10 ta nolmas", "k20": "20 ta nolmas",
           "err": "xato (L2)", "best": "eng katta n ta koeffitsient (oracle)",
           "cs": "n ta tasodifiy o'lchov + (P1)", "lin": "birinchi n ta koeffitsient (chiziqli)",
           "axis": "o'q bo'ylab: radius 1", "diag": "diagonal bo'ylab: radius 0.707",
           "ball": "L1 shari (romb)"},
    "en": {"true": "true x", "l1": "min L1 (P1): exact recovery", "l2": "min L2: spread out",
           "idx": "coordinate index", "val": "value",
           "n": "number of measurements n", "succ": "fraction recovered exactly",
           "k5": "5 nonzeros", "k10": "10 nonzeros", "k20": "20 nonzeros",
           "err": "error (L2)", "best": "largest n coefficients (oracle)",
           "cs": "n random measurements + (P1)", "lin": "first n coefficients (linear)",
           "axis": "along an axis: radius 1", "diag": "along the diagonal: radius 0.707",
           "ball": "L1 ball (diamond)"},
}
T = LABELS[LANG]
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b66"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
rng = np.random.default_rng(0)


def random_matrix(n, m):
    """n x m matritsa, ustunlari birlik sferada tekis taqsimlangan (norma 1)."""
    Phi = rng.standard_normal((n, m))
    return Phi / np.linalg.norm(Phi, axis=0)


def solve_p1(Phi, y):
    """(P1):  min ||theta||_1  shart: Phi theta = y   (linear program, theta = u - v)."""
    m = Phi.shape[1]
    res = linprog(np.ones(2 * m), A_eq=np.hstack([Phi, -Phi]), b_eq=y, bounds=(0, None))
    return res.x[:m] - res.x[m:]


def solve_l2(Phi, y):
    """Minimal L2-norma yechim (pseudo-inverse)."""
    return Phi.T @ np.linalg.solve(Phi @ Phi.T, y)


# ------------------------------------------------------------------
print("=== A) l_p 'norma':  ||theta||_p = (sum |theta_i|^p)^(1/p) ===")
def lp(theta, p):
    return np.sum(np.abs(theta) ** p) ** (1 / p)
spike = np.array([1.0, 0, 0, 0])
flat = np.array([0.5, 0.5, 0.5, 0.5])
for p in [2, 1, 0.5]:
    print(f"p = {p}:  spike (1,0,0,0) -> {lp(spike, p):.3f}    flat (0.5,0.5,0.5,0.5) -> {lp(flat, p):.3f}")

# ------------------------------------------------------------------
print("\n=== B) N-term approximation: theta_k = k^(-1/p), p = 1/2, m = 1000 ===")
p, m = 0.5, 1000
theta = np.arange(1, m + 1) ** (-1 / p)                 # 1, 1/4, 1/9, ...
print("||theta||_p =", round(lp(theta, p), 1), "  (R)")
for N in [10, 20, 40]:
    err = np.linalg.norm(theta[N:])                      # eng katta N tasini saqlasak qolgan xato
    print(f"N = {N:3d}:  xato = {err:.5f}    N^(1/2-1/p) = {N ** (0.5 - 1 / p):.5f}")
print("xato(10)/xato(20) =", round(np.linalg.norm(theta[10:]) / np.linalg.norm(theta[20:]), 2),
      "  (2^1.5 =", round(2 ** 1.5, 2), ")")

# ------------------------------------------------------------------
print("\n=== C) Haar transform, m = 4 ===")
r = 1 / np.sqrt(2)
Psi = np.array([[.5, .5, r, 0], [.5, .5, -r, 0], [.5, -.5, 0, r], [.5, -.5, 0, -r]])  # ustunlar = bazis
x = np.array([3.0, 3.0, 1.0, 1.0])
print("x =", x, "  theta = Psi^T x =", Psi.T @ x, "  Psi theta =", Psi @ (Psi.T @ x))

# ------------------------------------------------------------------
print("\n=== D) Gel'fand width: L1 shari (romb) R^2 da, n = 1 o'lchov ===")
for a, name in [(np.array([1.0, 0.0]), "a=(1,0)  -> nullspace = e2 o'qi"),
                (np.array([1.0, -1.0]), "a=(1,-1) -> nullspace = diagonal (1,1)")]:
    null = np.array([-a[1], a[0]])                       # a ga perpendikulyar yo'nalish
    pt = null / np.abs(null).sum()                       # rombning shu yo'nalishdagi chegarasi
    print(f"{name}:  romb bilan kesishma nuqtasi {pt},  radius = {np.linalg.norm(pt):.3f}")
for (mm, nn) in [(1000, 100), (1000, 400), (10**6, 10**4)]:
    print(f"m={mm}, n={nn}:  sqrt(log(m/n)/n) = {np.sqrt(np.log(mm / nn) / nn):.4f}")

# ------------------------------------------------------------------
print("\n=== E) 2x3 matritsa: (P1) nimani tiklaydi ===")
Phi23 = np.array([[1.0, 0.0, r], [0.0, 1.0, r]])
for x0 in [np.array([0, 0, 1.414]), np.array([1.0, 0, 0]), np.array([1.0, 1.0, 0])]:
    y = Phi23 @ x0
    print("x0 =", x0, " y =", y, " (P1) ->", solve_p1(Phi23, y))
ydual = np.array([r, r])
print("dual sertifikat y=(0.707,0.707): Phi^T y =", Phi23.T @ ydual)
print("x0=(1,1,0) uchun kerak y=(1,1):  Phi^T y =", Phi23.T @ np.array([1.0, 1.0]), " (uchinchisi > 1: bo'lmaydi)")

# ------------------------------------------------------------------
print("\n=== F) Asosiy tajriba: m = 256, n = 64, 8 ta nolmas ===")
m, n, k = 256, 64, 8
Phi = random_matrix(n, m)
x0 = np.zeros(m)
supp = rng.choice(m, k, replace=False)
x0[supp] = rng.choice([-1.0, 1.0], k) * rng.uniform(1, 2, k)
y = Phi @ x0
x1, x2 = solve_p1(Phi, y), solve_l2(Phi, y)
print("||x1 - x0||_2 =", f"{np.linalg.norm(x1 - x0):.2e}", "   ||x2 - x0||_2 =", round(np.linalg.norm(x2 - x0), 3))
print("nolmaslar: x0 =", k, "  x1 =", int(np.sum(np.abs(x1) > 1e-6)), "  x2 =", int(np.sum(np.abs(x2) > 1e-6)))
print("N log m = 8 * log(256) =", round(8 * np.log(256), 1))

fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for ax, v, title, col in zip(axes, [x0, x1, x2], [T["true"], T["l1"], T["l2"]], ["black", BLUE, GRAY]):
    ax.vlines(np.arange(m), 0, v, color=col, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=12); ax.set_xlabel(T["idx"])
axes[0].set_ylabel(T["val"])
plt.tight_layout(); plt.savefig(FIG + f"cs_demo_{LANG}.pdf"); plt.close()

# ------------------------------------------------------------------
print("\n=== G) Aniq tiklash ulushi (m = 200, 40 tajriba) ===")
m, trials = 200, 40
ns = np.arange(10, 121, 10)
rates = {}
for k in [5, 10, 20]:
    rates[k] = []
    for n in ns:
        ok = 0
        for _ in range(trials):
            Phi = random_matrix(n, m)
            x0 = np.zeros(m); x0[rng.choice(m, k, replace=False)] = rng.standard_normal(k)
            ok += np.linalg.norm(solve_p1(Phi, Phi @ x0) - x0) < 1e-5
        rates[k].append(float(ok) / trials)
    print(f"k = {k:2d}:", dict(zip(ns.tolist(), rates[k])))
fig, ax = plt.subplots(figsize=(6.2, 3.3))
for k, col, mk, lab in [(5, BLUE, "o", T["k5"]), (10, ORANGE, "s", T["k10"]), (20, AQUA, "^", T["k20"])]:
    ax.plot(ns, rates[k], marker=mk, color=col, linewidth=2, ms=6, label=lab)
ax.set_xlabel(T["n"] + "  (m = 200)"); ax.set_ylabel(T["succ"]); ax.set_ylim(-0.05, 1.08)
ax.legend(frameon=False, fontsize=10, loc="lower right")
plt.tight_layout(); plt.savefig(FIG + f"success_{LANG}.pdf"); plt.close()

# ------------------------------------------------------------------
print("\n=== H) Compressible signal: |theta|_(k) = k^(-2)  (p = 1/2), m = 512 ===")
m = 512
mags = np.arange(1, m + 1) ** (-2.0)
theta = np.zeros(m)
theta[rng.permutation(m)] = mags * rng.choice([-1.0, 1.0], m)   # joylari va ishoralari tasodifiy
ns = [32, 64, 96, 128, 192, 256]
e_best, e_cs, e_lin = [], [], []
for n in ns:
    e_best.append(np.linalg.norm(mags[n:]))                      # oracle: eng katta n tasi
    e_lin.append(np.linalg.norm(theta[n:]))                      # birinchi n ta koordinata
    Phi = random_matrix(n, m)
    e_cs.append(np.linalg.norm(solve_p1(Phi, Phi @ theta) - theta))
    print(f"n = {n:3d}:  oracle = {e_best[-1]:.2e}   CS+(P1) = {e_cs[-1]:.2e}   chiziqli = {e_lin[-1]:.2e}"
          f"   CS xatosi = oracle(N) bilan teng bo'lgan N ~ {int(np.searchsorted(-np.array([np.linalg.norm(mags[j:]) for j in range(m)]), -e_cs[-1]))}")
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.semilogy(ns, e_lin, marker="^", color=GRAY, linewidth=2, label=T["lin"])
ax.semilogy(ns, e_cs, marker="o", color=BLUE, linewidth=2, label=T["cs"])
ax.semilogy(ns, e_best, marker="s", color=ORANGE, linewidth=2, label=T["best"])
ax.set_xlabel("n  (m = 512)"); ax.set_ylabel(T["err"])
ax.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
plt.tight_layout(); plt.savefig(FIG + f"compressible_{LANG}.pdf"); plt.close()

# ------------------------------------------------------------------
print("\n=== I) CS1 va CS2 sonlarda: n = 50, m = 200, |J| = 5, 2000 ta tasodifiy J ===")
n, m = 50, 200
Phi = random_matrix(n, m)
smin, ratio = [], []
for _ in range(2000):
    J = rng.choice(m, 5, replace=False)
    smin.append(np.linalg.svd(Phi[:, J], compute_uv=False).min())
    v = Phi[:, J] @ rng.standard_normal(5)
    ratio.append(np.abs(v).sum() / (np.sqrt(n) * np.linalg.norm(v)))
print("CS1: eng kichik singular qiymat: min =", round(min(smin), 3), " o'rtacha =", round(np.mean(smin), 3))
print("CS2: ||v||_1 / (sqrt(n) ||v||_2): min =", round(min(ratio), 3), " o'rtacha =", round(np.mean(ratio), 3),
      "  (sqrt(2/pi) =", round(np.sqrt(2 / np.pi), 3), ")")
e1 = np.zeros(n); e1[0] = 1
print("taqqoslash: 'spike' vektor e1 uchun nisbat =", round(np.abs(e1).sum() / (np.sqrt(n) * np.linalg.norm(e1)), 3))
bad = np.column_stack([Phi[:, 0], Phi[:, 0] + 0.01 * Phi[:, 1]])
print("deyarli bir xil ikki ustun: eng kichik singular qiymat =", round(np.linalg.svd(bad, compute_uv=False).min(), 4))

# ------------------------------------------------------------------
print("\n=== J) Shovqinga barqarorlik (m = 256, n = 64, 8 ta nolmas) ===")
m, n, k = 256, 64, 8
Phi = random_matrix(n, m)
x0 = np.zeros(m); x0[rng.choice(m, k, replace=False)] = rng.choice([-1.0, 1.0], k) * rng.uniform(1, 2, k)
z = rng.standard_normal(n); z /= np.linalg.norm(z)
for eps in [0.0, 0.001, 0.01, 0.1]:
    xh = solve_p1(Phi, Phi @ x0 + eps * z)
    print(f"||shovqin||_2 = {eps:5.3f}:  ||x_hat - x0||_2 = {np.linalg.norm(xh - x0):.4f}")

# ------------------------------------------------------------------
print("\n=== K) CS1 bizning Lorenz library uchun (10 ta ustun, normallangan) ===")
def lorenz(s, t):
    return [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]]
X = odeint(lorenz, [-8, 8, 27], np.arange(0, 10, 0.001))
xx, yy, zz = X.T
Theta = np.column_stack([np.ones_like(xx), xx, yy, zz, xx**2, xx*yy, xx*zz, yy**2, yy*zz, zz**2])
Theta /= np.linalg.norm(Theta, axis=0)
names = ['1', 'x', 'y', 'z', 'x^2', 'xy', 'xz', 'y^2', 'yz', 'z^2']
for size in [2, 3, 4, 6, 10]:
    best = min(((np.linalg.svd(Theta[:, list(J)], compute_uv=False).min(), J)
                for J in combinations(range(10), size)), key=lambda t: t[0])
    print(f"|J| = {size:2d}:  eng kichik singular qiymat = {best[0]:.4f}   eng yomon J = {[names[j] for j in best[1]]}")
G = np.abs(Theta.T @ Theta); np.fill_diagonal(G, 0)
i, j = np.unravel_index(G.argmax(), G.shape)
print("mutual coherence =", round(G.max(), 4), " juftlik:", names[i], names[j])

# ------------------------------------------------------------------
print("\n=== L) Stylized applications: m^(1/4) log^(5/2)(m) ===")
for mm in [10**4, 10**6]:
    print(f"m = {mm}:  m^(1/2) = {mm**0.5:.0f},  m^(1/3) = {mm**(1/3):.0f},  m^(1/4) = {mm**0.25:.1f},"
          f"  m^(1/4) log^(5/2) m = {mm**0.25 * np.log(mm)**2.5:.0f}")
s = np.cumsum(1 / np.arange(1, 1001))
print("weak-l1 misol: sum_{k<=10} 1/k =", round(s[9], 2), " sum_{k<=1000} 1/k =", round(s[999], 2))

# ------------------------------------------------------------------
# Rasm: Gel'fand width, romb
fig, ax = plt.subplots(figsize=(4.4, 3.6))
ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color="black", linewidth=1.5, label=T["ball"])
ax.plot([0, 0], [-1, 1], color=ORANGE, linewidth=3, label=T["axis"])
ax.plot([-0.5, 0.5], [-0.5, 0.5], color=BLUE, linewidth=3, label=T["diag"])
ax.axhline(0, color=GRAY, linewidth=0.5); ax.axvline(0, color=GRAY, linewidth=0.5)
ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.25, 1.9); ax.grid(False)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(frameon=False, fontsize=9, loc="upper center")
plt.tight_layout(); plt.savefig(FIG + f"width_{LANG}.pdf"); plt.close()
print("Rasmlar saqlandi / figures saved.")
