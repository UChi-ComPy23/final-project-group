import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking
from src.Core.problems import ProblemBase

"""
Nested FISTA
Minimization model: minimize φ(f(x)) + λ g(Ax)

Assumptions:
-φ is Lipschitz, non-decreasing, and proximable
-f is smooth
-g is proper, closed, proximable
-λ > 0

Oracles:
-φ(x), prox_{αφ}(x)
-f(x), ∇f(x)
-g(x), prox_{αg}(x)
-A(x), A^T(y)
"""

class NestedFISTA(SolverBase):
    """Nested FISTA outer loop."""

    def __init__(self, problem, x0, lam, inner_solver, M0=1.0):
        super().__init__(problem, x0)
        self.x = x0.copy()
        self.y = x0.copy()
        self.t = 1.0
        self.t_prev = 1.0
        self.x_prev = x0.copy()

        self.lam = lam
        self.M = float(M0)
        self.inner_solver = inner_solver

    def step(self):
        """
        1) z_k = inner_solver(problem, y_k, λ, M)
        2) monotone acceptance
        3) momentum update
        4) extrapolation
        """
        # 1) inner step
        try:
            z_k = self.inner_solver(self.problem, self.y, self.lam, self.M)
        except TypeError:
            z_k = self.inner_solver(self.problem, self.y, self.lam, self.M, n_inner=20)

        # objective
        def F(x):
            Ax = self.problem.A(x)
            return self.problem.phi(self.problem.f(x)) + self.lam * self.problem.g(Ax)

        # 2) monotonicity rule
        x_new = z_k if F(z_k) <= F(self.x) else self.x.copy()

        # 3) momentum
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * self.t * self.t))

        # 4) extrapolation
        y_new = (
            x_new
            + (self.t / t_new) * (z_k - x_new)
            + (self.t_prev / t_new) * (x_new - self.x)
        )

        # update state
        self.x_prev = self.x
        self.x = x_new
        self.t_prev = self.t
        self.t = t_new
        self.y = y_new

        self.record(obj=F(self.x))
		
def accurate_inner_solver(problem, y, lam, M, n_inner=20):
    """
    More accurate inner solver for NestedFISTA.

    NestedFISTA must approximately solve
        prox_{(λ/M) g ∘ A}(y)
    which rarely has a closed form.

    We solve instead the surrogate:
        min_x  (M/2)||x - y||² + λ g(Ax)

    whose smooth part has gradient M(x - y).
    """

    z = y.copy()

    step = 1.0 / M

    for _ in range(n_inner):
        # gradient of (M/2)||x - y||²
        grad_smooth = M * (z - y)

        # forward-backward step
        z_tilde = z - step * grad_smooth
        z = problem.prox_g(z_tilde, step * lam)

    return z

