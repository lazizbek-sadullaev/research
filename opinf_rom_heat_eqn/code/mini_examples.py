# mini_examples.py
# Small stand-alone examples of every numpy function used in opinf_heat_demo.py.
# Each example is run, and the code + its literal output are written to
# ../note/generated/mini_examples.tex, so the note always shows real output.
# Run:  python mini_examples.py

import io, os, contextlib

examples = [
("np.arange(start, stop)", "Integers from start up to stop-1. We use it for the grid index i = 1,...,7.",
"""import numpy as np
i = np.arange(1, 8)
print(i)
print(i/8)"""),

("np.ones(n) and np.zeros((m, n))", "Arrays filled with ones or zeros. zeros((m, n)) reserves space for the snapshot matrix.",
"""import numpy as np
print(np.ones(5))
print(np.zeros((2, 5)))"""),

("np.diag(v, k)", "Puts the vector v on the k-th diagonal of a square matrix (k = 0 main, k = 1 above, k = -1 below).",
"""import numpy as np
print(np.diag([-2, -2, -2, -2, -2]))
print(np.diag([1, 1, 1, 1], 1))"""),

("Building the 1D Laplacian", "The sum of the three diagonals gives the tridiagonal matrix of Section 6 (here N = 5, without the 1/dx**2 factor).",
"""import numpy as np
N = 5
T = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
print(T)"""),

("np.eye(n)", "The identity matrix. A = I + dt L is one explicit Euler step.",
"""import numpy as np
print(np.eye(5))"""),

("The @ operator", "Matrix times vector (or matrix times matrix). One FOM step is A @ x.",
"""import numpy as np
T = np.diag(-2*np.ones(5)) + np.diag(np.ones(4), 1) + np.diag(np.ones(4), -1)
x = np.array([0., 1., 2., 1., 0.])
print(T @ x)"""),

("Filling columns: X[:, k]", "X[:, k] is column k of X. The time loop writes the state x(k) into column k.",
"""import numpy as np
X = np.zeros((3, 4))
X[:, 0] = [1., 2., 3.]
for k in range(3):
    X[:, k+1] = 0.5 * X[:, k]
print(X)"""),

("Slicing: M[:, :-1] and M[:, 1:]", "All columns except the last, and all columns except the first. Side by side they pair x(k) with x(k+1).",
"""import numpy as np
M = np.arange(10).reshape(2, 5)
print(M)
print(M[:, :-1])
print(M[:, 1:])"""),

("Transpose .T", "Swaps rows and columns. OpInf stacks the reduced states as ROWS of D, so we transpose.",
"""import numpy as np
M = np.arange(6).reshape(2, 3)
print(M)
print(M.T)"""),

("np.linalg.svd(X, full_matrices=False)", "X = U diag(s) Wt. Columns of U are the POD modes, s are the singular values (largest first). A rank-1 example: only one singular value is nonzero.",
"""import numpy as np
np.set_printoptions(precision=4, suppress=True)
X = np.outer([1., 2., 2.], [1., 0.5, 0.25, 0.125, 0.0625])
U, s, Wt = np.linalg.svd(X, full_matrices=False)
print(s)
print(U[:, 0])"""),

("U[:, :r]", "The first r columns: the POD basis V.",
"""import numpy as np
U = np.arange(15).reshape(5, 3)
print(U[:, :2])"""),

("np.linalg.lstsq(D, R, rcond=None)", "Finds the M that makes the residual D M - R as small as possible (least squares). Here the data come from x(k+1) = 0.8 x(k) exactly, so the fitted slope is 0.8.",
"""import numpy as np
x = 0.8**np.arange(6)            # 1, 0.8, 0.64, ...
D = x[:-1].reshape(-1, 1)        # column of x_k
R = x[1:].reshape(-1, 1)         # column of x_{k+1}
M, res, rank, sv = np.linalg.lstsq(D, R, rcond=None)
print(M)"""),

("np.linalg.norm and np.linalg.eigvals", "The (Frobenius) norm measures the size of a difference; eigvals gives the eigenvalues of a matrix.",
"""import numpy as np
np.set_printoptions(precision=4, suppress=True)
B = np.array([[0.9, 0.1], [0.1, 0.7]])
print(np.linalg.norm(B - np.diag([0.9, 0.7])))
print(np.linalg.eigvals(B))"""),
]

def esc(s):
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("^", "\\^{}")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "note", "generated")
os.makedirs(out_dir, exist_ok=True)
parts = []
for title, text, code in examples:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    parts.append("\\paragraph{\\texttt{%s}} %s\n" % (esc(title), esc(text)))
    parts.append("\\begin{lstlisting}[style=py]\n%s\n\\end{lstlisting}\n" % code)
    parts.append("\\begin{lstlisting}[style=out]\n%s\\end{lstlisting}\n" % buf.getvalue())
with open(os.path.join(out_dir, "mini_examples.tex"), "w") as f:
    f.write("\n".join(parts))
print("wrote", os.path.join(os.path.abspath(out_dir), "mini_examples.tex"))
