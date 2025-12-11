import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking, backtracking_composite
from src.Core.problems import ProblemBase

"""
Method: Proximal Gradient (Forward–Backward Splitting)
Minimization model: minimize f(x) + g(x)

Assumptions:
- f is smooth with Lipschitz-continuous gradient
- g is proper, closed, and proximable
- prox_g must be available through the problem interface

The method performs:
1) Forward (gradient) step on f: x̃ = x - α ∇f(x)
2) Backward (proximal) step on g: x⁺ = prox_{α g}(x̃)

Step size α may be fixed or selected by backtracking line search.
"""

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
            alpha = backtracking_composite(f=self.problem.f, grad_f=self.problem.grad,g_fun=self.problem.g,prox_g=self.problem.prox_g,x=self.x)
        else:
            alpha = self.alpha

        # Add safeguard for step size to prevent overflow
        if alpha > 1e10:  # If step size is too large, cap it
            alpha = 1.0

        # forward-backward update
        x_new = self.x - alpha * g
        # apply prox_g only if available
        prox_g = getattr(self.problem, "prox_g", None)
        if prox_g is not None and callable(prox_g):
            try:
                x_new = prox_g(x_new, alpha)
            except NotImplementedError:
                pass

        self.x = x_new

        # record objective
        obj = self.problem.f(self.x)
        self.objs.append(obj)
        self.record(obj=obj, x=self.x.copy())