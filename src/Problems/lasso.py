from src.Core.problems import ProblemBase
from src.util.prox_ops import prox_l1

"""
Concrete problem.
LASSO: f(x) = 1/2 ||A x - y||^2, g(x) = lambda * ||x||_1
"""

class LassoProblem(ProblemBase):
    """
    Lasso problem: least squares with L1 penalty.
    """
    def __init__(self, A, y, lam):
        self.A = A
        self.y = y
        self.lam = lam

    def f(self, x):
        """
        Smooth part f(x) = 0.5 * ||Ax - y||^2
        """
        r = self.A @ x - self.y
        return 0.5 * (r @ r)

    def grad(self, x):
        """
        Gradient of f(x)
        """
        return self.A.T @ (self.A @ x - self.y)

    def prox_g(self, x, alpha):
        """
        Proximal operator of g(x) = lam * ||x||_1
        """
        return prox_l1(x, alpha * self.lam)

