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
        grad_loss = -(self.A.T @ (self.y * s))
        grad_reg = self.lam * x
        return grad_loss + grad_reg

    def subgrad(self, x):
        """
        Smooth function, so subgradient = gradient
        """
        return self.grad(x)


class LogisticBoxProblem(ProblemBase):
    """
    Logistic regression with L2 regularization and box constraint:
        minimize  sum log(1 + exp(-y_i * a_i^T x)) + (lam/2)||x||^2
        subject to  lower <= x <= upper
    """

    def __init__(self, A, y, lam=0.0, lower=0.0, upper=1.0):
        self.A = A
        self.y = y
        self.lam = lam
        self.lower = lower
        self.upper = upper

    def f(self, x):
        z = self.A @ x
        loss = np.log(1 + np.exp(-self.y * z)).sum()
        reg = 0.5 * self.lam * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        z = self.A @ x
        s = 1 / (1 + np.exp(self.y * z))   # sigmoid(-y*z)
        grad_loss = -(self.A.T @ (self.y * s))
        grad_reg  = self.lam * x
        return grad_loss + grad_reg

    def subgrad(self, x):
        return self.grad(x)

    # ---- constraints for COMD ----
    def constraints(self, x):
        """
        Returns g(x) <= 0 violations.
        For box:   x - upper <= 0
                   lower - x <= 0
        """
        return np.concatenate([x - self.upper, self.lower - x])

    def constraint_subgrad(self, x, i):
        """Subgradient of violated constraint."""
        n = x.size
        if i < n:  # x_i - upper <= 0
            e = np.zeros(n)
            e[i] = 1.0
            return e
        else:      # lower - x_j <= 0
            j = i - n
            e = np.zeros(n)
            e[j] = -1.0
            return e

    def proj_X(self, x):
        return np.clip(x, self.lower, self.upper)


