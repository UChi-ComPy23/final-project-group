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
            + (self.t_prev / t_new) * (x_new - self.x))

        # update state
        self.x_prev = self.x
        self.x = x_new
        self.t_prev = self.t
        self.t = t_new
        self.y = y_new

        self.record(obj=F(self.x))
		
def accurate_inner_solver(problem, y, lam, M, n_inner=20, rho=None):
    """
    ADMM-based inner solver for NestedFISTA.

    Approximate: prox_{(λ/M) g ∘ A}(y)= argmin_x (M/2) ||x - y||^2 + λ g(Ax).

    We solve
        min_{x,z} (M/2)||x - y||^2 + λ g(z)
        s.t.      z = A x
    with ADMM on (x,z,u).
    """
    # penalty parameter
    if rho is None:
        rho = M  # simple M

    # try to get explicit A; otherwise fall back to matvecs
    if hasattr(problem, "Amat"):
        A_mat = problem.Amat
        m, n = A_mat.shape
        AtA = A_mat.T @ A_mat
        K = M * np.eye(n) + rho * AtA
        # pre-factorize K once per inner solve
        L = np.linalg.cholesky(K)
        def solve_K(rhs):
            # solve K x = rhs via Cholesky
            w = np.linalg.solve(L, rhs)
            return np.linalg.solve(L.T, w)
        A = lambda x: A_mat @ x
        AT = lambda v: A_mat.T @ v
    else:
        # generic matvec-only version (slower; uses AT(Ax))
        # assume x has same shape as y
        n = y.shape[0]
        def A(x):
            return problem.A(x)
        def AT(v):
            return problem.AT(v)
        def K_mv(x):
            return M * x + rho * AT(A(x))
        def solve_K(rhs, iters=50, tol=1e-8):
            # simple CG for SPD system K x = rhs
            x = np.zeros_like(rhs)
            r = rhs - K_mv(x)
            p = r.copy()
            rs_old = np.dot(r, r)
            for _ in range(iters):
                Ap = K_mv(p)
                alpha = rs_old / np.dot(p, Ap)
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = np.dot(r, r)
                if np.sqrt(rs_new) < tol:
                    break
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new
            return x

    # initialize ADMM variables
    x = y.copy()
    z = A(x)
    u = np.zeros_like(z)

    # ADMM iterations
    for _ in range(n_inner):
        # x-update: solve (M I + ρ AᵀA)x = M y + ρ Aᵀ(z - u)
        rhs = M * y + rho * AT(z - u)
        x = solve_K(rhs)

        # z-update: prox on g
        v = A(x) + u
        # here τ = λ / ρ → prox_{(λ/ρ) g}
        z = problem.prox_g(v, lam / rho)

        # dual update
        u = u + A(x) - z

    return x

