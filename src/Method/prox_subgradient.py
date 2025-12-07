import numpy as np
from src.Core.solver import SolverBase
from src.Core.problems import ProblemBase


"""
Method: proximal subgradient
Minimization model: minimize f(x) + λ g(x)

Assumptions:
- f is Lipschitz continuous (or at least has a subgradient / gradient)
- g is proper, closed, and proximable
- λ > 0

Oracles required from `problem`:
- f(x)
- either subgrad(x) or grad(x)
- (optional) g(x)
- prox_g(x, alpha)  for g
"""


class ProxSubgradient(SolverBase):
    """Proximal subgradient method."""

    def __init__(self, problem, x0, step_rule=None):
        """
        Parameters
        ----------
        problem : ProblemBase
            Provides f, (subgrad or grad), prox_g.
        x0 : np.ndarray
            Initial point.
        step_rule : callable or None
            Function k ↦ α_k specifying step size at iteration k.
            If None, use α_k = 1 / sqrt(k+1).
        """
        super().__init__(problem, x0)
        self.step_rule = step_rule

    # ------------------------------------------------------------------
    # helper: choose subgradient implementation
    # ------------------------------------------------------------------
    def _get_subgrad(self, x):
        """
        Prefer a real override of subgrad(x); if not provided,
        fall back to grad(x) if available.
        """
        # check whether subgrad is overridden in the concrete Problem
        if type(self.problem).subgrad is not ProblemBase.subgrad:
            return self.problem.subgrad(x)

        # otherwise try grad (many smooth problems只实现 grad)
        if type(self.problem).grad is not ProblemBase.grad:
            return self.problem.grad(x)

        raise RuntimeError(
            "ProxSubgradient needs either subgrad(x) or grad(x) "
            "implemented in the Problem class."
        )

    # ------------------------------------------------------------------
    # one proximal subgradient step
    # ------------------------------------------------------------------
    def step(self):
        """Perform one proximal subgradient update."""
        k = self.k

        # 1) choose step size
        if self.step_rule is None:
            alpha = 1.0 / np.sqrt(k + 1.0)
        else:
            alpha = float(self.step_rule(k))

        # 2) get a (sub)gradient of f at current x
        s = self._get_subgrad(self.x)

        # 3) gradient / subgradient step on f
        x_tilde = self.x - alpha * s

        # 4) prox step on g  (if prox_g is provided)
        if hasattr(self.problem, "prox_g"):
            x_new = self.problem.prox_g(x_tilde, alpha)
        else:
            # no g-part: pure subgradient on f
            x_new = x_tilde

        # update iterate
        self.x = x_new

        # 5) record diagnostics
        obj = None
        if hasattr(self.problem, "f"):
            if hasattr(self.problem, "g"):
                obj = float(self.problem.f(self.x) + self.problem.g(self.x))
            else:
                obj = float(self.problem.f(self.x))

        grad_norm = float(np.linalg.norm(s))

        # history["x"] 用来后面自己重算 f+λ‖x‖1
        self.record(
            obj=obj,
            x=self.x.copy(),
            step_size=alpha,
            grad_norm=grad_norm,
        )
