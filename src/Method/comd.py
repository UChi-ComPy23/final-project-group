import numpy as np
from src.Core.solver import SolverBase

"""
Co-mirror descent (Euclidean version)

Model:
    minimize   f(x)
    subject to g_i(x) <= 0   for i = 1,...,m
               x ∈ X

Idea:
    - If x^k is (approximately) feasible:  move along a subgradient of f.
    - Else: take a step against a subgradient of the *most violated* constraint.
    - After the step, project back to X (Euclidean projection).

Required oracles from `problem`:
    - f(x)
    - subgrad(x)                 # a subgradient of f at x
    - g(x)                       # vector of constraint values g_i(x)
    - subgrad_g(x, i)            # subgradient of g_i at x
    - proj_X(x)                  # projection onto X

Notes:
    This is written in Euclidean geometry; a true "co-mirror" method would use
    a general Bregman distance. Here we keep the interface simple and push the
    geometry into `proj_X` if needed.
"""


class CoMirrorDescent(SolverBase):
    """Co-mirror descent method (Euclidean version)."""

    def __init__(
        self,
        problem,
        x0,
        step_size=None,
        feas_tol=1e-4,
        grad_tol=1e-4,
    ):
        """
        Parameters
        ----------
        problem : ProblemBase subclass
            Must implement: f, subgrad, g, subgrad_g, proj_X.
        x0 : ndarray
            Initial *feasible* point (ideally satisfies g_i(x0) <= 0 and x0 ∈ X).
        step_size : float or callable, optional
            If float, use a constant step size.
            If callable, it should be step_size(k) -> α_k at iteration k.
            Default: 1 / sqrt(k+1).
        feas_tol : float
            Tolerance on constraint violation used in stop().
        grad_tol : float
            Tolerance on (sub)gradient norm used in stop().
        """
        super().__init__(problem, x0)

        # step-size schedule
        if step_size is None:
            # default diminishing step size: 1 / sqrt(k+1)
            self._step_size_fun = lambda k: 1.0 / np.sqrt(k + 1.0)
        elif callable(step_size):
            self._step_size_fun = step_size
        else:
            alpha_const = float(step_size)
            self._step_size_fun = lambda k, a=alpha_const: a

        self.feas_tol = float(feas_tol)
        self.grad_tol = float(grad_tol)

    # ------------------------------------------------------------------
    # one iteration
    # ------------------------------------------------------------------
    def step(self):
        """
        Perform one co-mirror descent update.

        If x^k is (approximately) feasible (max_i g_i(x^k) <= 0),
        use f-subgradient:
            d_k ∈ ∂f(x^k).

        Else choose most violated constraint:
            i_k = argmax_i g_i(x^k),
            d_k ∈ ∂g_{i_k}(x^k).

        Then do a projected subgradient step:
            x^{k+1} = proj_X( x^k - α_k d_k ).
        """
        xk = self.x
        k = self.k

        # constraint values
        g_vals = np.asarray(self.problem.g(xk))
        max_violation = float(np.max(g_vals)) if g_vals.size > 0 else 0.0

        # choose direction: either f-subgrad or constraint subgrad
        if max_violation <= 0.0:
            # (approximately) feasible => descend on f
            d = np.asarray(self.problem.subgrad(xk))
            mode = "obj"
        else:
            # infeasible => fix the most violated constraint
            i_max = int(np.argmax(g_vals))
            d = np.asarray(self.problem.subgrad_g(xk, i_max))
            mode = "constr"

        # step size
        alpha_k = self._step_size_fun(k)

        # projected subgradient step
        x_temp = xk - alpha_k * d
        x_new = self.problem.proj_X(x_temp)

        # update state
        self.x = x_new

        # diagnostics
        grad_norm = float(np.linalg.norm(d))
        obj_val = float(self.problem.f(x_new))
        g_new = np.asarray(self.problem.g(x_new))
        max_viol_new = float(np.max(g_new)) if g_new.size > 0 else 0.0

        self.record(
            obj=obj_val,
            step_size=alpha_k,
            grad_norm=grad_norm,
            feas_violation=max_viol_new,
            mode=mode,        # "obj" or "constr"
            x=x_new.copy(),
        )

    # ------------------------------------------------------------------
    # stopping rule
    # ------------------------------------------------------------------
    def stop(self):
        """
        Simple stopping rule based on:
            - max constraint violation <= feas_tol
            - (sub)gradient norm <= grad_tol

        If you prefer to run for a fixed number of iterations, you can ignore
        this and just control `max_iter` in `run()`.
        """
        if "feas_violation" not in self.history:
            return False

        feas = self.history["feas_violation"][-1]
        grad_norm = self.history.get("grad_norm", [np.inf])[-1]

        return (feas <= self.feas_tol) and (grad_norm <= self.grad_tol)
	
