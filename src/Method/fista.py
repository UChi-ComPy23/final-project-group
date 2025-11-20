from core.solver import SolverBase
from core.util.linesearch import backtracking
import numpy as np

class FISTA(SolverBase):
    """FISTA method: accelerate proximal gradient
    """
    def __init__(self, problem, x0, alpha=None, monotone=False):
        super().__init__(problem, x0)
        self.alpha = alpha
        self.monotone = monotone

        # FISTA-specific variables
        self.y = x0.copy()
        self.t = 1.0
        self.obj_prev = self.problem.f(self.x)

    def step(self):
        g = self.problem.grad(self.y)
        p = -g

        # Step size: fixed or backtracking
        if self.alpha is None:
            f = self.problem.f
            alpha = backtracking(f, self.y, p, g)
        else:
            alpha = self.alpha

        # FISTA forward-backward update
        x_new = self.y - alpha * g
        x_new = self.problem.prox_g(x_new, alpha)

        # Monotone variant (optional)
        f_new = self.problem.f(x_new)
        if self.monotone and f_new > self.obj_prev:
            # restart to keep objective decreasing
            x_new = self.x.copy()
            self.y = x_new.copy()
            self.t = 1.0
        else:
            # Standard FISTA momentum update
            t_new = (1 + np.sqrt(1 + 4*self.t*self.t)) / 2.0
            self.y = x_new + (self.t - 1)/t_new * (x_new - self.x)
            self.t = t_new

        # Update iterate and objective
        self.x = x_new
        self.obj_prev = f_new
        self.record(obj=f_new)
