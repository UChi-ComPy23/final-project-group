from core.solver import SolverBase

"""
smoothed FISTA
Minimization model: minimize f(x) + λ_g g(Ax) + λ_h h(x)

Assumptions:
-f smooth
-g is proper, closed, proximable
-h is proper, closed, proximable
-λ_g > 0, λ_h > 0

Oracles:
- f(x) & ∇f(x)
- g(x) & prox_{αg}(x)
- h(x) & prox_{αh}(x)
- A(x) & A^T(y)
"""

class SmoothedFISTA(SolverBase):
    """Smoothed FISTA method
	"""
    def __init__(self, problem, x0):
        """
        problem: ProblemBase 
        x0: initial point.
        """
        super().__init__(problem, x0)

    def step(self):
        """Perform one smoothed FISTA update
		"""
        raise NotImplementedError
