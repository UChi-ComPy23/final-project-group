import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking
from src.Core.problems import ProblemBase

"""
Method: proximal subgradient
Minimization model: minimize f(x) + λ g(x)

Assumptions:
-f is Lipschitz continuous
-g is proper, closed, and proximable
-λ > 0

Update rule:
x̃ = x^k - α_k * s_k where s_k ∈ ∂f(x^k)
x^{k+1} = prox_{α_k g}(x̃)

Typical step sizes: α_k = O(1 / √k) for subgradient-type convergence
	
Oracles:
-f(x)
-f'(x): a subgradient of f
-g(x)
-prox_{αg}(x): proximal operator of g
"""

class ProxSubgradient(SolverBase):
    """Proximal subgradient method."""

    def __init__(self, problem, x0, step_rule=None):
        """
        Parameters
        ----------
        problem: f, (subgrad or grad), prox_g.
        x0: Initial point.
        step_rule: Function k ↦ α_k specifying the step size.
            If None, uses α_k = 1 / sqrt(k+1), standard diminishing rule.
        """
        super().__init__(problem, x0)
        self.step_rule = step_rule

    # helper: pick subgrad() if implemented; otherwise use grad()
    def _get_subgrad(self, x):
        # subgrad overridden
        if type(self.problem).subgrad is not ProblemBase.subgrad:
            return self.problem.subgrad(x)

        # otherwise try grad
        if type(self.problem).grad is not ProblemBase.grad:
            return self.problem.grad(x)

        raise RuntimeError(
            "ProxSubgradient needs either subgrad(x) or grad(x) "
            "implemented in the Problem class.")

    def step(self):
        """Perform one proximal subgradient update
		"""
        k = self.k
		
        # 1) choose step size 
        if self.step_rule is None:
            alpha = 1.0 / np.sqrt(k + 1.0)
        else:
            alpha = float(self.step_rule(k))

        # 2) obtain (sub)gradient of f
        s = self._get_subgrad(self.x)

        # 3) forward/subgradient step on f 
        x_tilde = self.x - alpha * s

        # 4) backward/prox step on g 
        if hasattr(self.problem, "prox_g"):
            x_new = self.problem.prox_g(x_tilde, alpha)
        else:
            x_new = x_tilde  # pure subgradient method

        # update iterate
        self.x = x_new

        # 5) diagnostics
        obj = None
        if hasattr(self.problem, "f"):
            if hasattr(self.problem, "g"):
                obj = float(self.problem.f(self.x) + self.problem.g(self.x))
            else:
                obj = float(self.problem.f(self.x))

        grad_norm = float(np.linalg.norm(s))

        self.record(
            obj=obj,
            x=self.x.copy(),
            step_size=alpha,
            grad_norm=grad_norm,)
