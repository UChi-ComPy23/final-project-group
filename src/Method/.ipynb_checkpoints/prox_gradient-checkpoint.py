from Core.solver import SolverBase
from Core.util.linesearch import backtracking

"""
Method: proximal gradient

Minimization model: minimize   f(x) + λ g(x)

Assumptions:
- f is smooth with Lipschitz gradient
- g is proper, closed, and proximable
- λ > 0

Oracles:
- f(x)
- ∇f(x): gradient of f
- g(x)
- prox_{αg}(x): proximal operator of g
"""

class ProxGradient(SolverBase):
    """Proximal gradient method
	"""
    def __init__(self, problem, x0, alpha=None, btls=True):
        """
        problem: ProblemBase providing desired oracles
        x0: initial point.
        alpha(step size): if None, use backtracking.
        btls:bool, whether to use backtracking line search.
        """
        super().__init__(problem, x0)
        self.alpha = alpha
        self.btls = btls

    def step(self):
        g = self.problem.grad(self.x)
        p = -g  #gradient descent direction

        if self.alpha is None:# use btls find step size
            f = self.problem.f 
            alpha = backtracking(f, self.x, p, g)
        else:
            alpha = self.alpha

        # forward-backward update
        x_new = self.x - alpha * g
        if hasattr(self.problem, "prox_g"):
            self.x = self.problem.prox_g(x_new, alpha)
        else:
            raise RuntimeError("missing prox_g ")

        self.record(obj=self.problem.f(self.x)) #record
		