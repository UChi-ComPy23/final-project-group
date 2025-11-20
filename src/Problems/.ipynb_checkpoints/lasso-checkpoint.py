from src.Core.problems import ProblemBase
from src.util.prox_ops import prox_l1
"""
Concrete problem.
LASSO:f(x) = 1/2 ||A x - y||^2, g(x) = lambda * ||x||_1
"""

class LassoProblem(ProblemBase):
    """Lasso
	"""
    def __init__(self, A, y, lam):
        self.A = A
        self.y = y
        self.lam = lam

    def f(self, x):
		"""f(x)
		"""
        r = self.A @ x - self.y
        return 1/2 * (r @ r)

    def grad(self, x):
        """gradient of f
		"""
        return self.A.T @ (self.A @ x - self.y)

    def prox_g(self, x, alpha):
        """Prox of g(x) 
		"""
        return prox_l1(x, alpha * self.lam)
