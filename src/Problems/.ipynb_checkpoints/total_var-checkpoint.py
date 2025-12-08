import numpy as np
from src.Core.problems import ProblemBase
from src.util.prox_ops import prox_l1

"""
Total Variation Denoising Problem: minimize 0.5 * ||x - b||^2 + lam * ||D x||_1
"""

class TotalVariationProblem(ProblemBase):
    """
    1D Total Variation Denoising with implicit forward difference operator.
    """
    def __init__(self, b, lam=0.1):
        """
        b: observed signal (n,)
        lam: TV regularization weight
        """
        self.b = b
        self.lam = lam
        self.n = len(b)

    def f(self, x):
        r = x - self.b
        return 0.5 * (r @ r)
		
    def prox_f(self, v, tau):
        """
        prox_{tau f}(v) where f(x) = 0.5 ||x - b||^2
        Solution: (v + tau b) / (1 + tau)
        """
        return (v + tau * self.b) / (1.0 + tau)

    def grad(self, x):
        return x - self.b

    def A(self, x):
        # forward difference: D x
        return x[1:] - x[:-1]

    def AT(self, y):
        """
        Adjoint of D:
        D^T y = [ -y1, y1 - y2, ..., y_{n-2} - y_{n-1}, y_{n-1} ]
        """
        n = len(y) + 1
        out = np.zeros(n)
        out[0] = -y[0]
        out[1:-1] = y[:-1] - y[1:]
        out[-1] = y[-1]
        return out

    def g(self, z):
        return self.lam * np.linalg.norm(z, 1)

    def prox_g(self, z, alpha):
        return prox_l1(z, alpha * self.lam)
