from src.core.solver import SolverBase
from src.util.linesearch import backtracking
import numpy as np

class ProxGradient(SolverBase):
    """Proximal Gradient (PG) method: x_{k+1} = prox_{αg}(x - α ∇f(x))"""

    def __init__(self, problem, x0, alpha=None):
        """
        problem: ProblemBase object providing f, grad, prox_g
        x0: initial point
        alpha: step size (if None → backtracking)
        """
        super().__init__(problem, x0)
        self.alpha = alpha  # fixed step or None (backtracking)

    def step(self):
        """Perform one standard proximal gradient step."""
        g = self.problem.grad(self.x)          # ∇f(x)
        p = -g                                # descent direction

        # Determine step size
        if self.alpha is None:
            f = self.problem.f
            alpha = backtracking(f, self.x, p, g)
        else:
            alpha = self.alpha

        # Forward-backward update
        x_new = self.x - alpha * g
        if hasattr(self.problem, "prox_g"):
            self.x = self.problem.prox_g(x_new, alpha)
        else:
            raise RuntimeError("Problem missing prox_g")

        # Record objective value
        self.record(obj=self.problem.f(self.x))
