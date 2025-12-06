import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking

class ProxGradient(SolverBase):
    """Proximal gradient method"""
    def __init__(self, problem, x0, alpha=None, btls=True, max_iter=1000):
        """
        problem: ProblemBase providing desired oracles
        x0: initial point.
        alpha(step size): if None, use backtracking.
        btls:bool, whether to use backtracking line search.
        max_iter: maximum number of iterations
        """
        super().__init__(problem, x0)
        self.alpha = alpha
        self.btls = btls
        self.max_iter = max_iter
        self.objs = [] #initialize objective history list 

    def step(self):
        """a step for prox_gradient method
        """
        g = self.problem.grad(self.x)
        p = -g  #gradient descent direction

        if self.alpha is None: # use btls find step size
            f = self.problem.f 
            alpha = backtracking(f, self.x, p, g)
        else:
            alpha = self.alpha

        # Add safeguard for step size to prevent overflow
        if alpha > 1e10:  # If step size is too large, cap it
            alpha = 1.0

        # forward-backward update
        x_new = self.x - alpha * g
        if hasattr(self.problem, "prox_g"):
            self.x = self.problem.prox_g(x_new, alpha)
        else:
            raise RuntimeError("missing prox_g ")

        # Record objective value
        current_obj = self.problem.f(self.x)
        self.objs.append(current_obj)
        if hasattr(self, 'record'):
            self.record(obj=current_obj)
