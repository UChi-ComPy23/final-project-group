from core.solver import SolverBase

"""
Method: proximal subgradient
Minimization model: minimize f(x) + λ g(x)

Assumptions:
-f is Lipschitz continuous
-g is proper, closed, and proximable
-λ > 0

Oracles:
-f(x)
-f'(x): a subgradient of f
-g(x)
-prox_{αg}(x): proximal operator of g
"""

class ProxSubgradient(SolverBase):
    """Proximal subgradient method
	"""
    def __init__(self, problem, x0, step_rule=None):
        """
        problem: ProblemBase providing desired oracles
        x0: initial point.
        step_rule: function k ↦ α_k specifying step-size at iteration k.
        """
        super().__init__(problem, x0)
        self.step_rule = step_rule

    def step(self):
        """Perform one proximal subgradient update
		"""
        raise NotImplementedError
