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

"""
Quadratic objective with many linear inequalities:
    minimize 0.5 ||A x - b||^2
    subject to G x <= h

Used to compare COMD (handles constraints) vs PG/PSG (no real feasibility).
"""

class PolytopeQuadraticProblem(ProblemBase):
    """Quadratic objective with a polytope constraint Gx <= h."""

    def __init__(self, A, b, G, h):
        """
        A, b : quadratic data (0.5‖Ax - b‖²)
        G, h : linear inequality constraints Gx <= h
        """
        self.A = A
        self.b = b
        self.G = G
        self.h = h

    def f(self, x):
        """Compute objective 0.5 * ||A x - b||^2."""
        r = self.A @ x - self.b
        return 0.5 * np.dot(r, r)

    def grad(self, x):
        """Gradient of 0.5 * ||A x - b||^2."""
        return self.A.T @ (self.A @ x - self.b)

    def subgrad(self, x):
        """Smooth problem ⇒ subgradient = gradient."""
        return self.grad(x)

    def constraints(self, x):
        """Return vector of constraint violations g(x) = Gx - h."""
        return self.G @ x - self.h

    def constraint_subgrad(self, x, i):
        """Subgradient of violated constraint g_i(x) = G[i]·x - h[i]."""
        return self.G[i]

    def proj_X(self, x):
        """
        Projection onto feasible set.
        (Left as identity — PG/PSG do not enforce feasibility.
         Only COMD uses constraints to remain feasible.)
        """
        return x
