from core.solver import SolverBase

"""
Alternating direction linearized proximal method of multipliers (ADLPMM)
Minimization model: minimize f(x) + λ g(Ax)

Assumptions:
-f is proper, closed, proximable
-g is proper, closed, proximable
-λ > 0

Oracles:
-f(x), prox_{αf}(x)
-g(x), prox_{αg}(x)
-A(x), A^T(y)
"""

class ADLPMM(SolverBase):
    """ADLPMM method, requires problems to implement prox_f and A, AT.
	"""
    def __init__(self, problem, x0, z0, u0, rho=1.0, abstol=1e-4, reltol=1e-2):
        """
        problem: ProblemBase
        x0, z0, u0: initial primal and dual variables
        rho: penalty parameter
        abstol: absolute tolerance for stopping
        reltol: relative tolerance for stopping
        """
        super().__init__(problem, x0)
        self.z = z0
        self.u = u0
        self.rho = rho
        self.abstol = abstol
        self.reltol = reltol

    def step(self):
        """Perform one ADLPMM iteration (x-update, z-update, u-update)
		"""
        raise NotImplementedError

