from core.solver import SolverBase

"""
Method: fast dual proximal gradient
Minimization model: minimize f(x) + λ g(Ax)

Assumptions:
-f is proper, closed, and strongly convex
-g is proper, closed, and proximable
-λ > 0

Oracles:
-f(x)
-∇f^*(x): gradient of the convex conjugate of f
-g(x), prox_{αg}(x)
-A(x), A^T(y)
"""

class FDPG(SolverBase):
    """Fast dual proximal gradient method
	"""
    def __init__(self, problem, x0):
        """
        problem: ProblemBase providing f, ∇f^*, g, prox_{αg}, and linear maps
        x0: initial point (dual-related variable depending on setup)
        """
        super().__init__(problem, x0)

    def step(self):
        """Perform one fast dual proximal gradient update
		"""
        raise NotImplementedError
