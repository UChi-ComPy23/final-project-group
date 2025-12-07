import numpy as np
from src.Core.solver import SolverBase

"""
Fast Dual Proximal Gradient (FDPG)

Primal model (our convention in this project):
    minimize    f(x) + g(Ax)

Assumptions:
    - f is proper, closed, and µ-strongly convex
    - g is proper, closed, proximable
    - A is a linear operator with adjoint A^T

Dual problem:
    minimize    φ(y) := f^*(-A^T y) + g^*(y)

We run an accelerated proximal-gradient (FISTA) on the dual variable y,
treating f^*(-A^T y) as smooth and g^*(y) as non-smooth.

Required oracles from `problem` (ProblemBase subclass):
    - f(x)
    - grad_conjugate(s)         # ∇ f^*(s)
    - g(z)          (optional, only for logging primal objective)
    - prox_g(z, α)  # prox_{α g}(z)
    - A(x), AT(y)
    - (optional) normA : spectral norm ‖A‖₂
    - (optional) mu   : strong convexity parameter of f

If L (Lipschitz constant of ∇_y f^*(-A^T y)) is not provided,
we try to use L = ‖A‖² / µ when `problem.normA` and `problem.mu` exist.
"""


class FDPG(SolverBase):
    """Fast Dual Proximal Gradient method (FISTA on the dual)."""

    def __init__(self, problem, x0, L=None):
        """
        Parameters
        ----------
        problem : ProblemBase
            Provides: f, grad_conjugate, prox_g, A, AT (and optionally g, normA, mu).
        x0 : np.ndarray
            Initial dual variable y⁰ (stored in self.x).
        L : float, optional
            Lipschitz constant of the smooth dual part:
                φ_s(y) = f^*(-A^T y)
            If None, we try L = normA**2 / mu using problem.normA and problem.mu.
        """
        super().__init__(problem, x0)

        # FISTA-style acceleration variables on the dual
        self.y = x0.copy()   # extrapolated dual variable
        self.t = 1.0         # FISTA momentum parameter

        # Determine Lipschitz constant L
        if L is not None:
            self.L = float(L)
        else:
            normA = getattr(problem, "normA", None)
            mu = getattr(problem, "mu", None)
            if (normA is not None) and (mu is not None):
                self.L = (normA ** 2) / float(mu)
            else:
                raise ValueError(
                    "FDPG needs a Lipschitz constant L for the dual gradient.\n"
                    "Either pass L explicitly, or let the problem provide "
                    "`normA` (‖A‖₂) and `mu` (strong convexity of f)."
                )

        # Step size α = 1/L
        self.alpha = 1.0 / self.L

    # ------------------------------------------------------------------
    # One FDPG iteration
    # ------------------------------------------------------------------
    def step(self):
        """
        Perform one FDPG iteration (FISTA on the dual):

        Let y^k be the extrapolated dual variable.
        1) Compute primal x(y^k) = ∇ f^*(-A^T y^k)
        2) Dual gradient of smooth part:
               ∇_y f^*(-A^T y) = -A x(y)
        3) Gradient step on the smooth part:
               v = y^k - α * ( -A x(y^k) )
        4) Prox-step on g^* via Moreau:
               y^{k+1} = prox_{α g^*}(v)
                        = v - α * prox_{(1/α) g}(v / α)
        5) FISTA acceleration on y.
        """
        # Current extrapolated dual variable
        y = self.y

        # ----- 1) primal x(y) = ∇ f^*(-A^T y) -----
        s = -self.problem.AT(y)                 # s = -A^T y
        x_primal = self.problem.grad_conjugate(s)

        # ----- 2) gradient of smooth dual part: -A x(y) -----
        grad_smooth = -self.problem.A(x_primal)

        # ----- 3) gradient step on smooth part -----
        v = y - self.alpha * grad_smooth

        # ----- 4) prox on g^* using Moreau identity -----
        # prox_{α g^*}(v) = v - α * prox_{(1/α) g}(v / α)
        sigma = self.alpha
        prox_input = v / sigma
        prox_g = self.problem.prox_g(prox_input, 1.0 / sigma)
        y_new = v - sigma * prox_g

        # ----- 5) FISTA acceleration -----
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * self.t ** 2)) / 2.0
        y_extrap = y_new + ((self.t - 1.0) / t_new) * (y_new - self.x)

        # Update state (self.x is the "current dual iterate")
        self.x = y_new
        self.y = y_extrap
        self.t = t_new

        # ----- Diagnostics: primal objective & dual variable -----
        Ax = self.problem.A(x_primal)
        obj = float(self.problem.f(x_primal))
        if hasattr(self.problem, "g"):
            obj += float(self.problem.g(Ax))

        self.record(
            obj=obj,
            x_primal=x_primal.copy(),
            y=self.x.copy(),
        )
