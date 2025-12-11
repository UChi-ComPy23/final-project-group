import numpy as np
from src.Core.problems import ProblemBase
from src.util.proj import proj_box

"""
Simple Quadratic Problem: f(x) = 0.5 x^T Q x + b^T x
Used for testing smooth solvers (PG, FISTA, etc.)
"""

class QuadraticProblem(ProblemBase):
    """simple quadratic problem"""
    
    def __init__(self, Q, b):
        self.Q = Q
        self.b = b

    def f(self, x):
        r = self.Q @ x
        return 0.5 * (x @ r) + self.b @ x

    def grad(self, x):
        return self.Q @ x + self.b

    def prox_g(self, x, alpha):
        """ Projection onto the box [0,1]^n for example"""
        return np.clip(x, 0, 1)


"""
Quadratic objective with box constraints:
f(x) = 0.5 x^T Q x + b^T x
lower <= x <= upper
Works with COMD, PG, PSG.
"""		
class BoxQuadraticProblem(ProblemBase):
    """
    f(x) = 0.5 x^T Q x + b^T x
    with box constraints lower <= x <= upper
    PG, PSG, COMD compatible
    """

    def __init__(self, Q, b, lower, upper):
        self.Q = Q
        self.b = b
        self.lower = lower
        self.upper = upper

    def f(self, x):
        return 0.5 * x @ (self.Q @ x) + self.b @ x

    def grad(self, x):
        return self.Q @ x + self.b

    #COMDspecific
    def constraints(self, x):
        """Return constraint violations g_i(x) <= 0."""
        return np.concatenate([x - self.upper, self.lower - x])

    def constraint_subgrad(self, x, i):
        """Subgradient of i-th constraint."""
        n = len(x)
        g = np.zeros(n)
        m = n  # number of upper constraints

        if i < m:
            g[i] = 1 # x_i - upper_i
        else:
            j = i - m
            g[j] = -1 # lower_j - x_j
        return g

    def proj_X(self, x):
        """Projection used by COMD"""
        return proj_box(x, self.lower, self.upper)

    # COMD: objective subgrad 
    def subgrad(self, x):
        return self.grad(x)