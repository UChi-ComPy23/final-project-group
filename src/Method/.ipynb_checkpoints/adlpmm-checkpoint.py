import numpy as np
from src.Core.solver import SolverBase
from src.util.linesearch import backtracking
from src.Core.problems import ProblemBase

"""
Alternating direction linearized proximal method of multipliers (ADLPMM)
Minimization model: minimize f(x) + λ g(Ax) subject to z = A x

Scaled ADMM form:
x-update  (linearized)
z-update  via prox_g
u-update  dual ascent on constraint z = A x

Assumptions:
- f is proper, closed, proximable
- g is proper, closed, proximable 
- λ > 0
- A is a linear operator with adjoint A^T

Oracles:
- f(x), prox_f(x, alpha)
- g(z), prox_g(z, alpha)
- A(x), AT(y)
"""

class ADLPMM(SolverBase):
    """ADLPMM method, requires problems to implement prox_f and A, AT."""
    def __init__(self, problem, x0, z0=None, u0=None, lam=1, rho=1, abstol=1e-4, reltol=1e-2):
        super().__init__(problem, x0)

        # compute Ax to determine dimension m
        Ax0 = self.problem.A(self.x)
        m = Ax0.shape[0]

        # initialize / validate z0
        if z0 is None:
            z0 = Ax0.copy()
        else:
            z0 = np.asarray(z0)
            if z0.shape[0] != m:
                raise ValueError(f"ADLPMM init: z0 has shape {z0.shape}, but Ax has dimension {m}.")

        # initialize / validate u0
        if u0 is None:
            u0 = np.zeros(m)
        else:
            u0 = np.asarray(u0)
            if u0.shape[0] != m:
                raise ValueError(f"ADLPMM init: u0 has shape {u0.shape}, but Ax has dimension {m}.")

        # finalize
        self.z = z0.copy()
        self.u = u0.copy()
        self.lam = lam
        self.rho = rho
        self.abstol = abstol
        self.reltol = reltol
	
    def step(self):
        """
        1) x-update (linearized, via prox_f)
        2) z-update (prox_g)
        3) u-update (dual variable)
        Augmented Lagrangian in scaled form:
            L_ρ(x,z,u) = f(x) + λ g(z) + (ρ/2) ||A x - z + u||^2 - (ρ/2)||u||^2

        Linearized x-update:
            x^{k+1} ≈ prox_{τ f}( x^k - τ ρ A^T( A x^k - z^k + u^k ) )
        z-update: 
            z^{k+1} = prox_{(λ/ρ) g}( A x^{k+1} + u^k )
        u-update:
            u^{k+1} = u^k + A x^{k+1} - z^{k+1}
        """
        if not hasattr(self.problem, "prox_f"):
            raise RuntimeError("ADLPMM requires problem.prox_f to be implemented.")
        if not hasattr(self.problem, "prox_g"):
            raise RuntimeError("ADLPMM requires problem.prox_g to be implemented.")

        rho = self.rho

        # 1)
        Ax = self.problem.A(self.x)
        r = Ax - self.z + self.u
        grad_x = self.problem.AT(r)

        # step size τ: 1/(ρ * ||A||^2) if normA is provided, else 1/ρ
        normA = getattr(self.problem, "normA", None)
        if normA is not None:
            tau = 1 / (rho * normA**2 + 1e-12)
        else:
            tau = 1 / rho

        # linearized gradient step then prox_f
        x_lin = self.x - tau * rho * grad_x
        x_new = self.problem.prox_f(x_lin, tau)

        # 2) z-update 
        Ax_new = self.problem.A(x_new)
        v = Ax_new + self.u
        # z^{k+1} = prox_{(λ/ρ) g}(v)
        z_new = self.problem.prox_g(v, self.lam / rho)

        # 3) u-update (dual variable)
        u_new = self.u + Ax_new - z_new

        # update state
        self.x = x_new
        self.z = z_new
        self.u = u_new

        # diagnostics
        obj = None
        if hasattr(self.problem, "f"):
            obj = float(self.problem.f(self.x))
            if hasattr(self.problem, "g"):
                obj += float(self.lam * self.problem.g(Ax_new))

        r_norm = float(np.linalg.norm(Ax_new - z_new))
        s_vec = rho * self.problem.AT(z_new - self.z)
        s_norm = float(np.linalg.norm(s_vec))

        self.record(
            obj=obj,
            r_norm=r_norm,
            s_norm=s_norm,
            x=self.x.copy(),
            z=self.z.copy(),)

    def stop(self):
        """stopping condition for ADLPMM. stops when:
            r_norm <= abstol + reltol * max(||A x||, ||z||)
            s_norm <= abstol + reltol * s_norm   
        """
        H = self.history
        if "r_norm" not in H or "s_norm" not in H:
            return False

        r_norm = H["r_norm"][-1]
        s_norm = H["s_norm"][-1]

        # primal tolerance
        Ax_norm = np.linalg.norm(self.problem.A(self.x))
        z_norm = np.linalg.norm(self.z)
        eps_pri = self.abstol + self.reltol * max(Ax_norm, z_norm)

        # dual tolerance (simplified scaling)
        eps_dual = self.abstol + self.reltol * s_norm

        return (r_norm <= eps_pri) and (s_norm <= eps_dual)
