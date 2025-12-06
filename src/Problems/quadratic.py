from src.Core.problems import ProblemBase

"""
Concrete problem.
Quadratic convex problem: f(x) = 1/2 x^T Q x + b^T x with Q symmetric PSD.
"""

class QuadraticProblem(ProblemBase):
    """
    the Quadratic type of problem
    """

    def __init__(self, Q, b):
        self.Q = Q
        self.b = b

    def f(self, x):
        """
        f(x) = 0.5 * x^T Q x + b^T x
        """
        return 0.5 * x @ (self.Q @ x) + self.b @ x

    def grad(self, x):
        """
        Gradient: Qx + b
        """
        return self.Q @ x + self.b

    def subgrad(self, x):
        """
        Smooth function, so subgradient = gradient
        """
        return self.grad(x)
