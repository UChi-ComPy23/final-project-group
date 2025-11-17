from core.solver import SolverBase

"""
Nested FISTA
Minimization model: minimize φ(f(x)) + λ g(Ax)

Assumptions:
-φ is Lipschitz, non-decreasing, and proximable
-f is smooth
-g is proper, closed, proximable
-λ > 0

Oracles:
-φ(x), prox_{αφ}(x)
-f(x), ∇f(x)
-g(x), prox_{αg}(x)
-A(x), A^T(y)
"""

class NestedFISTA(SolverBase):
    """Nested FISTA method
	"""
    def __init__(self, problem, x0):
        """
        problem: ProblemBase 
        x0: initial point.
        """
        super().__init__(problem, x0)

    def step(self):
        """Perform one nested FISTA update
		"""
        raise NotImplementedError
