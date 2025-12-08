import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking
from src.Core.problems import ProblemBase
"""
Method: Smoothed FISTA (Moreau envelope on g ∘ A)
Minimization model: minimize f(x) + λ_g * e_μ g(Ax) + λ_h * h(x)
					where e_μ g is the Moreau envelope of g:  e_μ g(z) = min_u g(u) + (1/(2μ)) ||u - z||²

Assumptions:
- f is smooth, grad(x) available
- g is proper, closed, proximable (prox_g available)
- h is proper, closed, proximable (prox_h available) – optional if λ_h = 0
- A is linear with adjoint AT
- μ > 0 smoothing parameter
- L > 0 Lipschitz constant of the smooth gradient:
      ∇ f(x) + λ_g * A^T ( (Ax - prox_{μ g}(Ax)) / μ )

oracles：
- f(x)
- grad(x)
- prox_g(z, alpha)
- A(x), AT(y)
- (optional) prox_h(x, alpha)
"""
class SmoothedFISTA(SolverBase):
    """Smoothed FISTA method (Moreau envelope on g ∘ A)
    """
    def __init__(self, problem, x0, mu, L, lambda_g=1.0, lambda_h=0.0, monotone=False):
        """
        problem: f, grad, prox_g, A, AT (and optionally prox_h).
        x0: initial point
        mu: smoothing parameter μ > 0 for Moreau envelope of g
        L: Lipschitz constant of smooth gradient part
        lambda_g: weight on e_μ g(Ax)
        lambda_h: weight on h(x)
        monotone: if True, enforce monotone FISTA (restart when objective increases)
        """
        super().__init__(problem, x0)
        self.mu = float(mu)
        self.L = float(L)
        self.lambda_g = float(lambda_g)
        self.lambda_h = float(lambda_h)
        self.monotone = monotone

        # FISTA variables
        self.y = x0.copy()  
        self.t = 1       
        self.objs = []   

        # initial objective if we can compute it
        obj0 = self._objective(self.x)
        self.objs.append(obj0)
        self.obj_prev = obj0

    def _smooth_grad(self, x):
        """ Compute gradient of the smooth part:
            ∇f(x) + λ_g * A^T ( (Ax - prox_{μ g}(Ax)) / μ )
        """
        # grad f
        g_f = self.problem.grad(x)

        # Moreau smoothed g(Ax)
        Ax = self.problem.A(x)
        prox_mu_g = self.problem.prox_g(Ax, self.mu) # prox_{μ g}(Ax)
        v = (Ax - prox_mu_g) / self.mu # ∇ e_μ g(Ax)

        g_g = self.lambda_g * self.problem.AT(v)

        return g_f + g_g

    def _objective(self, x):
        """
        Try to compute smoothed objective:
            f(x) + λ_g * e_μ g(Ax) + λ_h * h(x)
        """
        fx = float(self.problem.f(x))

        # Moreau envelope term, only if g and prox_g exist
        val_g = 0
        if hasattr(self.problem, "prox_g") and hasattr(self.problem, "g"):
            Ax = self.problem.A(x)
            u = self.problem.prox_g(Ax, self.mu) # u* = prox_{μ g}(Ax)
            g_u = float(self.problem.g(u))
            # e_μ g(Ax) = g(u) + (1/(2μ))||u - Ax||²
            diff = u - Ax
            moreau = g_u + (0.5 / self.mu) * float(diff @ diff)
            val_g = self.lambda_g * moreau

        # h term if we have it
        val_h = 0
        if self.lambda_h != 0 and hasattr(self.problem, "h"):
            val_h = self.lambda_h * float(self.problem.h(x))

        return fx + val_g + val_h

    def step(self):
        """
        Perform one Smoothed FISTA iteration:
        1) Compute gradient of smooth part at y^k
        2) Gradient step: x̃ = y^k - (1/L) * ∇F_μ(y^k)
        3) Prox step on λ_h h (if prox_h exists)
        4) FISTA momentum update
        5) (optional) monotone restart if objective increases
        """
        # 1) 
        g = self._smooth_grad(self.y)

        # 2) 
        alpha = 1 / self.L
        x_tilde = self.y - alpha * g

        # 3)
        if self.lambda_h != 0 and hasattr(self.problem, "prox_h"):
            x_new = self.problem.prox_h(x_tilde, alpha * self.lambda_h)
        else:
            x_new = x_tilde

        # 4) 
        t_new = (1 + np.sqrt(1 + 4 * self.t**2)) / 2
        y_new = x_new + ((self.t - 1.0) / t_new) * (x_new - self.x)

        # 5) 
        if self.monotone:
            f_new = self._objective(x_new)
            if f_new > self.obj_prev:
                # restart: no momentum
                x_new = x_tilde
                if self.lambda_h != 0 and hasattr(self.problem, "prox_h"):
                    x_new = self.problem.prox_h(x_tilde, alpha * self.lambda_h)
                y_new = x_new.copy()
                t_new = 1.0
                f_new = self._objective(x_new)
            self.obj_prev = f_new
        else:
            f_new = self._objective(x_new)
            self.obj_prev = f_new

        # update state
        self.x = x_new
        self.y = y_new
        self.t = t_new

        # record objective history
        self.objs.append(f_new)
        self.record(obj=f_new, x=self.x.copy())
