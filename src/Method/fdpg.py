import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking
from src.Core.problems import ProblemBase

"""
Method: Fast Dual Proximal Gradient (FDPG)
Minimization model: minimize f(x) + g(Ax)

Assumptions:
- f is proper, closed, and µ-strongly convex
- g is proper, closed, proximable
- A is a linear operator with adjoint A^T

Dual problem: minimize φ(y) := f^*(-A^T y) + g^*(y)

We run an accelerated proximal-gradient (FISTA) on the dual variable y,
treating f^*(-A^T y) as smooth and g^*(y) as non-smooth.

If L (Lipschitz constant of ∇_y f^*(-A^T y)) is not provided,
we try to use L = ‖A‖² / µ when `problem.normA` and `problem.mu` exist.
"""

class FDPG(SolverBase):
    """
	Fast Dual Proximal Gradient method (FISTA on the dual)
	"""

    def __init__(self, problem, x0, L=None):
        """
        problem: Provides: f, grad_conjugate, prox_g, A, AT (and optionally g, normA, mu).
        x0: Initial dual variable y^0 (stored in self.x).
        L: Lipschitz constant of the smooth dual part φ_s(y) = f^*(-A^T y)
            If None, we try L = normA**2 / mu using problem.normA and problem.mu.
        """
        super().__init__(problem, x0)

        self.y = x0.copy()# extrapolated dual variable
        self.t = 1 # FISTA momentum parameter
        self.objs = []

        # Determine Lipschitz constant L
        if L is not None:
            self.L = float(L)
        else:
            normA = getattr(problem, "normA", None)
            mu = getattr(problem, "mu", None)
            if (normA is not None) and (mu is not None):
                self.L = (normA ** 2) / float(mu)
            else:
                raise ValueError("FDPG needs a Lipschitz constant L for the dual gradient.")
				
        self.alpha = 1 / self.L # Step size α = 1/L

    def step(self):
        """
        Perform one FDPG iteration (FISTA on the dual):
        Let y^k be the extrapolated dual variable.
        1) Compute primal x(y^k) = ∇ f^*(-A^T y^k)
        2) Dual gradient of smooth part:∇_y f^*(-A^T y) = -A x(y)
        3) Gradient step on the smooth part: v = y^k - α * ( -A x(y^k) )
        4) Prox-step on g^* via Moreau: y^{k+1} = prox_{α g^*}(v) = v - α * prox_{(1/α) g}(v / α)
        5) FISTA acceleration on y.
        """
        y = self.y

        # 1) 
        s = -self.problem.AT(y) # s = -A^T y
        x_primal = self.problem.grad_conjugate(s)

        # 2)
        grad_smooth = -self.problem.A(x_primal)

        # 3) 
        v = y - self.alpha * grad_smooth

        # 4) 
        sigma = self.alpha
        prox_input = v / sigma
        prox_g = self.problem.prox_g(prox_input, 1 / sigma)
        y_new = v - sigma * prox_g

        # 5) 
        t_new = (1 + np.sqrt(1 + 4 * self.t ** 2)) / 2
        y_extrap = y_new + ((self.t - 1) / t_new) * (y_new - self.x)

        # Update state (self.x is the "current dual iterate")
        self.x = y_new
        self.y = y_extrap
        self.t = t_new

        # primal objective & dual variable
        Ax = self.problem.A(x_primal)
        obj = float(self.problem.f(x_primal))
        if hasattr(self.problem, "g"):
            obj += float(self.problem.g(Ax))

        self.objs.append(obj)
        self.record(
            obj=obj,
            x_primal=x_primal.copy(),
            y=self.x.copy(),)
