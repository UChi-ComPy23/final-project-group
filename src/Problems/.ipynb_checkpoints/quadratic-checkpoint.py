from src.Core.problems import ProblemBase

"""
Concrete problem.
Quadratic convex problem:f(x) = 1/2 x^T Q x + b^T x with Q symmetric PSD.
"""

class QuadraticProblem(ProblemBase):

    def __init__(self, Q, b):
        self.Q = Q
        self.b = b

    def f(self, x):
		"""f(x)
		"""
        return 0.5 * x @ (self.Q @ x) + self.b @ x

    def grad(self, x):
		"""gradient, Qx+b
		"""
        return self.Q @ x + self.b

    def subgrad(self, x):
		"""smooth function so grad = subgrad.
		"""
        return self.grad(x)