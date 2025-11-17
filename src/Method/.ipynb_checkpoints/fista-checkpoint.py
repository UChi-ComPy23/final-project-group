from core.solver import SolverBase

"""
Method: FISTA
Minimization model: minimize f(x) + λ g(x)

Assumptions:
-f is smooth
-g is proper, closed, and proximable
-λ > 0

Oracles:
- f(x), ∇f(x)
- g(x), prox_{αg}(x)
"""

class FISTA(SolverBase):
    """FISTA method
	"""
    def __init__(self, problem, x0, alpha=None, monotone=False):
        """
        problem: ProblemBase providing desired oracles
        x0: initial point
        alpha(step size): if None, use backtracking.
        monotone:bool, whether to enforce monotone variant of FISTA.
        """
        super().__init__(problem, x0)
        self.alpha = alpha
        self.monotone = monotone

    def step(self):
        """Perform one FISTA iteration
		"""
        raise NotImplementedError
