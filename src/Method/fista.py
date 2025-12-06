import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking

class FISTA(SolverBase):
    """FISTA method: accelerate proximal gradient"""

    def __init__(self, problem, x0, alpha=None, monotone=False):
        super().__init__(problem, x0)
        self.alpha = alpha
        self.monotone = monotone

        # FISTA-specific variables
        self.y = x0.copy()
        self.t = 1.0
        self.obj_prev = self.problem.f(self.x)
        self.objs = [self.obj_prev]

    def step(self):
        g = self.problem.grad(self.y)
        p = -g

        # Step size: fixed or backtracking
        if self.alpha is None:
            f = self.problem.f
            alpha = backtracking(f, self.y, p, g)
        else:
            alpha = self.alpha

        # proximal gradient step at y
        x_new = self.y + alpha * p
        if hasattr(self.problem, "prox_g"):
            x_new = self.problem.prox_g(x_new, alpha)
        else:
            raise RuntimeError("missing prox_g")

        # update momentum
        t_new = (1 + np.sqrt(1 + 4 * self.t**2)) / 2
        y_new = x_new + ((self.t - 1) / t_new) * (x_new - self.x)

        # monotone version
        if self.monotone:
            f_new = self.problem.f(x_new)
            if f_new > self.obj_prev:
                # restart: undo momentum
                x_new = self.x + alpha * p
                if hasattr(self.problem, "prox_g"):
                    x_new = self.problem.prox_g(x_new, alpha)
                y_new = x_new.copy()
                t_new = 1.0
            self.obj_prev = f_new

        # update state
        self.x = x_new
        self.t = t_new
        self.y = y_new

        # record objective
        current_obj = self.problem.f(self.x)
        self.objs.append(current_obj)
        if hasattr(self, "record"):
            self.record(obj=current_obj)

