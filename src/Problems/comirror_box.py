import numpy as np
from src.Core.problems import ProblemBase
from src.util.proj import proj_box

"""
Nonnegative Least Squares: minimize 0.5 * ||A x - b||^2   subject to x >= 0
"""

class CoMirrorBoxProblem(ProblemBase):
    """Box-constrained least squares
	"""

    def __init__(self, A, b, l, u):
        self.A = A
        self.b = b
        self.l = l
        self.u = u

    def f(self, x):
        """Smooth part f(x)"""
        r = self.A @ x - self.b
        return 0.5 * (r @ r)

    def grad(self, x):
        """Gradient of f(x)"""
        return self.A.T @ (self.A @ x - self.b)

    def subgrad(self, x):
        """Since f is smooth, its subgradient is its gradient
		"""
        return self.grad(x)

    def g(self, x):
        """Constraint function g(x)
		"""
        return np.concatenate([x - self.u, self.l - x])

    def subgrad_g(self, x, i):
        """Subgradient of g_i(x)
		"""
        n = len(x)
        if i < n:
            return np.eye(n)[i]
        return -np.eye(n)[i - n]

    def proj_X(self, x):
        """Projection onto constraint set
		"""
        return proj_box(x, self.l, self.u)
