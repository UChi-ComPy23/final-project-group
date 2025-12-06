import numpy as np
from src.Core.problems import ProblemBase

"""
Constrained least squares: minimize 0.5 * ||A x - b||^2   subject to x in C
where C is defined via a projection operator.
"""

class ConstrainedLSProblem(ProblemBase):
    """
    Least-squares objective with a convex constraint set,
    implemented via its projection (prox of indicator).
    """

    def __init__(self, A, b, projection):
        """
        A: matrix (m x n)
        b: vector (m,)
        projection: function proj(x) returning projection onto C
        """
        self.A = A
        self.b = b
        self.projection = projection

    def f(self, x):
        """
        f(x) = 0.5 * ||A x - b||^2
        """
        r = self.A @ x - self.b
        return 0.5 * (r @ r)

    def grad(self, x):
        """
        Gradient: A^T (A x - b)
        """
        return self.A.T @ (self.A @ x - self.b)

    def prox_g(self, x, alpha):
        """
        Prox of indicator function = projection onto C
        """
        return self.projection(x)

    def subgrad(self, x):
        """
        Smooth, so subgradient = gradient
        """
        return self.grad(x)
