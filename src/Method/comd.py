from core.solver import SolverBase

"""
Method: co-mirror descent
Minimization model:bminimize f(x) subject to g_i(x) ≤ 0,  x ∈ X

Assumptions:
-f is Lipschitz continuous
-each g_i is Lipschitz
-X is a simple convex set (simplex/ball/box/spectahedron)

Oracles:
- f(x)
- f'(x): subgradient of f
- g_i(x)
- g_i'(x): subgradient of g_i
"""

class CoMirrorDescent(SolverBase):
    """Co-mirror descent method
	"""
    def __init__(self, problem, x0):
        """
        problem: ProblemBase providing desired oracles and feasible set X
        x0: initial feasible point.
        """
        super().__init__(problem, x0)

    def step(self):
        """Perform one co-mirror descent update
		"""
        raise NotImplementedError
