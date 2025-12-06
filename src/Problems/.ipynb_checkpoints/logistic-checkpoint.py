import numpy as np
from src.Core.problems import ProblemBase

"""
Concrete problem.
Binary logistic regression:
    f(x) = sum_i log(1 + exp(-y_i * (a_i^T x))) + (lam/2) * ||x||^2
where y_i ∈ {+1, -1}.
"""

class LogisticProblem(ProblemBase):
    """
    Logistic regression with optional L2 regularization
    """

    def __init__(self, A, y, lam=0.0):
        """
        A: data matrix (m x n)
        y: labels in {+1, -1}
        lam : L2 regularization parameter
        """
        self.A = A
        self.y = y
        self.lam = lam

    def f(self, x):
        """
        Logistic loss: sum log(1 + exp(-y_i * a_i^T x)) + (lam/2)||x||^2
        """
        z = self.A @ x
        loss = np.log(1 + np.exp(-self.y * z)).sum()
        reg = 0.5 * self.lam * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        """
        Gradient of logistic loss:-sum y_i * a_i * sigmoid(-y_i * a_i^T x) + lam * x
        """
        z = self.A @ x
        s = 1 / (1 + np.exp(self.y * z)) # sigmoid(-y z)
        grad_loss = -(self.y * s) @ self.A
        grad_reg = self.lam * x
        return grad_loss + grad_reg

    def subgrad(self, x):
        """
        Smooth function, so subgradient = gradient
        """
        return self.grad(x)

