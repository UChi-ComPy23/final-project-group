import numpy as np
from src.Core.problems import ProblemBase

"""
Simple Quadratic Problem: f(x) = 0.5 x^T Q x + b^T x
Used for testing smooth solvers (PG, FISTA, etc.)
"""

class QuadraticProblem(ProblemBase):
    def __init__(self, Q, b):
        self.Q = Q
        self.b = b

    def f(self, x):
        r = self.Q @ x
        return 0.5 * (x @ r) + self.b @ x

    def grad(self, x):
        return self.Q @ x + self.b
