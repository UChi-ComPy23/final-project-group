import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking
from src.Core.problems import ProblemBase

"""
Method: co-mirror descent
Minimization model:bminimize f(x) subject to g_i(x) ≤ 0,  x ∈ X

Idea:
- If x^k is (approximately) feasible:  move along a subgradient of f.
- Else: take a step against a subgradient of the *most violated* constraint.
- After the step, project back to X (Euclidean projection).
	
Assumptions:
-f is Lipschitz continuous
-each g_i is Lipschitz
-X is a simple convex set (simplex/ball/box/spectahedron)

Oracles:
- f(x)
- f'(x): subgradient of f
- g_i(x)
- g_i'(x): subgradient of g_i

Notes:
This is written in Euclidean geometry; a true "co-mirror" method would use
a general Bregman distance. Here we keep the interface simple and push the
geometry into `proj_X`.
"""

class CoMirrorDescent(SolverBase):
    """
    Co-mirror descent method (Euclidean version)
    """

    def __init__(self, problem, x0, step_size=None, feas_tol=1e-4, grad_tol=1e-4):
        """
        problem : must supply f, subgrad, g, subgrad_g, proj_X
        x0 : initial point (ideally feasible)
        step_size : float or callable; default is 1/sqrt(k+1)
        feas_tol : tolerance for constraint violation
        grad_tol : tolerance for gradient norm
        """
        super().__init__(problem, x0)

        # step-size schedule
        if step_size is None:
            self._step_size_fun = lambda k: 1 / np.sqrt(k + 1)
        elif callable(step_size):
            self._step_size_fun = step_size
        else:
            a = float(step_size)
            self._step_size_fun = lambda k, a=a: a

        self.feas_tol = float(feas_tol)
        self.grad_tol = float(grad_tol)
        self.objs = []

    def step(self):
        """
        Perform one co-mirror descent iteration:
        1) Evaluate constraint violation
        2) Choose direction: objective subgrad or constraint subgrad
        3) Compute step size
        4) Subgradient step then projection: x^{k+1} = proj_X(x^k - α_k d_k)
        """
        xk = self.x
        k = self.k

        #  1)
        if hasattr(self.problem, "constraints"):
            g_vals = np.asarray(self.problem.constraints(xk))
        elif hasattr(self.problem, "g"): # CoMirrorBoxProblem
            g_vals = np.asarray(self.problem.g(xk))
        else:
            g_vals = np.zeros(1)

        max_violation = float(np.max(g_vals)) if g_vals.size > 0 else 0

        # 2) 
        if max_violation <= 0:
            d = np.asarray(self.problem.subgrad(xk)) # objective subgradient
            mode = "obj"
        else:
            i_max = int(np.argmax(g_vals))
            if hasattr(self.problem, "constraint_subgrad"):
                d = np.asarray(self.problem.constraint_subgrad(xk, i_max))
            else:
                d = np.asarray(self.problem.subgrad_g(xk, i_max)) # legacy method
            mode = "constr"

        # 3) step size
        alpha_k = self._step_size_fun(k)

        #  4) projection
        x_temp = xk - alpha_k * d
        if hasattr(self.problem, "proj_X"):
            x_new = self.problem.proj_X(x_temp)
        else:
            x_new = self.problem.proj(x_temp)  

        # update state
        self.x = x_new

        # diagnostics
        grad_norm = float(np.linalg.norm(d))
        obj_val = float(self.problem.f(x_new))

        if hasattr(self.problem, "constraints"):
            g_new = np.asarray(self.problem.constraints(x_new))
        else:
            g_new = np.asarray(self.problem.g(x_new))

        max_viol_new = float(np.max(g_new)) if g_new.size > 0 else 0

        self.objs.append(obj_val)
        self.record(
            obj=obj_val,
            step_size=alpha_k,
            grad_norm=grad_norm,
            feas_violation=max_viol_new,
            mode=mode,
            x=x_new.copy(),)

    def stop(self):
        """stop when
        - max constraint violation <= feas_tol
        - gradient norm <= grad_tol
        """
        if "feas_violation" not in self.history:
            return False

        feas = self.history["feas_violation"][-1]
        grad_norm = self.history.get("grad_norm", [np.inf])[-1]

        return (feas <= self.feas_tol) and (grad_norm <= self.grad_tol)

