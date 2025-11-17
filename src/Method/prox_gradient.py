from core.solver import SolverBase

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
    def __init__(self, problem, x0, alpha=None, backtracking=True):
        """
        problem: ProblemBase providing desired oracles
        x0: initial point.
        alpha(step size): if None, use backtracking.
        backtracking:bool, whether to use backtracking line search.
        """
        super().__init__(problem, x0)
        self.alpha = alpha
        self.backtracking = backtracking

    def step(self):
        """Perform one proximal gradient update
		"""
        raise NotImplementedError
