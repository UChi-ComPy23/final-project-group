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
#FISTA
def fista(f, grad_f, prox_g, x0, L, max_iter=500):
    x = x0.copy()
    y = x.copy()
    t = 1.0
    objs = []
    for k in range(max_iter):
        x_new = prox_g(y - (1.0/L)*grad_f(y), 1.0/L)
        t_new = (1 + np.sqrt(1 + 4*t*t)) / 2.0
        y = x_new + (t-1)/t_new * (x_new - x)
        x, t = x_new, t_new
        objs.append(f(x))
    return x, np.array(objs)
