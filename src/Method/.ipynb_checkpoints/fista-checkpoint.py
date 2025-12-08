import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking, backtracking_composite
from src.Core.problems import ProblemBase

"""
Method: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
Minimization model: minimize f(x) + g(x)

Assumptions:
- f is smooth (∇f available)
- g is proper, closed, and proximable (prox_g available)
- Step size α is either fixed or obtained via backtracking line search

FISTA performs:
1) Gradient step on f
2) Proximal step on g
3) Nesterov acceleration using momentum t_k

Optional:
- Monotone FISTA variant: ensures objective does not increase by restarting momentum
"""

class FISTA(SolverBase):
    """FISTA method: accelerated proximal gradient
	"""

    def __init__(self, problem, x0, alpha=None, monotone=False):
        """
        problem: ProblemBase subclass providing f, grad, and prox_g
        x0: initial point
        alpha: fixed step size (if None, use backtracking)
        monotone: use monotone FISTA (restart when objective increases)
        """
        super().__init__(problem, x0)
        self.alpha = alpha
        self.monotone = monotone

        self.y = x0.copy() # extrapolated point
        self.t = 1 # momentum parameter
        self.obj_prev = self.problem.f(self.x)
        self.objs = [self.obj_prev] # store objective history

    def step(self):
        """Perform one FISTA iteration: gradient → prox → momentum update."""
        g = self.problem.grad(self.y) # gradient at extrapolated point
        p = -g # descent direction

        # Step size: fixed or backtracking
        if self.alpha is None:
            f = self.problem.f
            alpha = backtracking_composite(f=self.problem.f,grad_f=self.problem.grad,g_fun=self.problem.g,prox_g=self.problem.prox_g,x=self.y)
        else:
            alpha = self.alpha

        # 1) Proximal gradient step
        x_new = self.y + alpha * p
        if hasattr(self.problem, "prox_g"):
            x_new = self.problem.prox_g(x_new, alpha)
        else:
            raise RuntimeError("missing prox_g")

        # 2) Update momentum parameter and extrapolated point
        t_new = (1 + np.sqrt(1 + 4 * self.t**2)) / 2
        y_new = x_new + ((self.t - 1) / t_new) * (x_new - self.x)

        # 3) Monotone FISTA: restart momentum if objective increases
        if self.monotone:
            f_new = self.problem.f(x_new)
            if f_new > self.obj_prev:
                # restart: remove momentum
                x_new = self.x + alpha * p
                if hasattr(self.problem, "prox_g"):
                    x_new = self.problem.prox_g(x_new, alpha)
                y_new = x_new.copy()
                t_new = 1.0
            self.obj_prev = f_new

        # update
        self.x = x_new
        self.t = t_new
        self.y = y_new

        # record 
        current_obj = self.problem.f(self.x)
        self.objs.append(current_obj)
        if hasattr(self, "record"):
            self.record(obj=current_obj)
