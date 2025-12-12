import numpy as np
from src.Core.problems import ProblemBase

"""
Composite Group Lasso
minimize f(x) + g(Ax)
    where:f(x) = 1/2 ||x - b||^2, g(z) = lam * sum_g ||z_g||_2
"""

class GroupLassoCompositeProblem(ProblemBase):
    """Composite Group Lasso in the form f(x) + g(Ax)."""

    def __init__(self, A, b, groups, lam):
        self._A = A # matrix for z = A x
        self._AT = A.T
        self.b = b # target for f(x)
        self.groups = groups # list of index arrays for groups in z
        self.lam = lam  

        # for methods that need spectral info / strong convexity
        self.normA = np.linalg.norm(A, 2)
        self.mu = 1                 

    def f(self, x):
        """Smooth fidelity term f(x) = (1/2)*||x - b||^2."""
        r = x - self.b
        return 0.5 * (r @ r)

    def grad(self, x):
        """Gradient of f: ∇f(x) = x - b."""
        return x - self.b

    def prox_f(self, v, tau):
        """
        Proximal operator of τ f at v:
            prox_{τ f}(v) = argmin_x τ * (1/2)||x - b||^2 + (1/2)||x - v||^2.
        Closed form: x = (v + τ b) / (1 + τ).
        """
        return (v + tau * self.b) / (1.0 + tau)

    # Conjugate gradient for FDPG
    def grad_conjugate(self, s):
        """
        Gradient of the conjugate f*(s).

        For f(x) = 1/2||x - b||^2, we have
            f*(s) = 1/2 ||s||^2 + s^T b,
        so ∇f*(s) = s + b.
        """
        return s + self.b

    def A(self, x):
        """Forward operator: z = A x."""
        return self._A @ x

    def AT(self, z):
        """Adjoint operator: A^T z."""
        return self._AT @ z

    # Nonsmooth part g(z) = lam * sum_g ||z_g||_2
    def g(self, z):
        """Group penalty g(z) = lam * Σ_g ||z_g||_2."""
        out = 0.0
        for g_idx in self.groups:
            out += np.linalg.norm(z[g_idx])
        return self.lam * out

    def prox_g(self, z, alpha):
        """
        Proximal operator of g on z = A x:
            prox_{α g}(z),
        implemented via block soft-thresholding for each group.
        """
        out = z.copy()
        tau = alpha * self.lam

        for g_idx in self.groups:
            v = out[g_idx]
            nrm = np.linalg.norm(v)
            if nrm <= tau:
                out[g_idx] = 0.0
            else:
                out[g_idx] = (1.0 - tau / nrm) * v

        return out