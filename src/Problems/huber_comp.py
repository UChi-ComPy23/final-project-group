import numpy as np
from src.Core.problems import ProblemBase
from src.util.prox_ops import prox_l1

"""
Huber composite model: minimize phi(f(x)) + lam ||x||_1.
f = Huber residual
g = L1 penalty
phi customizable
"""
class HuberCompositeProblem(ProblemBase):
    """
    Composite model
    This model is appropriate for NestedFISTA, which requires
    a nested structure phi(f(x)) and a nonsmooth term g(Ax).
    """

    def __init__(self, A, b, delta_f, lam=0.1, phi_mode="sqrt"):
        self.Amat = A
        self.b = b
        self.delta_f = delta_f
        self.lam = lam
        self.phi_mode = phi_mode
        self.eps = 1e-6  # used for sqrt/log numerical stability

    def f(self, x):
        """smooth part f"""
        r = self.Amat @ x - self.b
        absr = np.abs(r)
        quad = absr <= self.delta_f
        val = np.sum(0.5 * r[quad] ** 2)
        val += np.sum(self.delta_f * (absr[~quad] - 0.5 * self.delta_f))
        return val

    def grad(self, x):
        """
        Gradient of phi(f(x)): ∇(phi∘f)(x) = phi'(f(x)) * ∇f(x)
        """
        # gradient of f(x)
        r = self.Amat @ x - self.b
        absr = np.abs(r)
        g = np.where(absr <= self.delta_f, r, self.delta_f * np.sign(r))
        grad_f = self.Amat.T @ g

        # scalar derivative phi'(f(x))
        u = self.f(x)
        return self.phi_grad(u) * grad_f

    def phi(self, u):
        """Scalar outer function applied to f(x)."""
        if self.phi_mode == "sqrt":
            return np.sqrt(u + self.eps)
        elif self.phi_mode == "log":
            return np.log1p(u)
        elif self.phi_mode == "identity":
            return u

    def phi_grad(self, u):
        """Derivative φ'(u) used for chain rule."""
        if self.phi_mode == "sqrt":
            return 1 / (2 * np.sqrt(u + self.eps))
        elif self.phi_mode == "log":
            return 1 / (1 + u)
        elif self.phi_mode == "identity":
            return 1

    def g(self, z):
        """Non-smooth term g(z) = ||z||_1."""
        return np.sum(np.abs(z))

    def prox_g(self, z, alpha):
        """
        Proximal operator of alpha * g(z):
            prox_{α||·||_1}(z) = soft-threshold(z, α)
        """
        return prox_l1(z, alpha)

    def A(self, x):
        """Apply A to x."""
        return self.Amat @ x

    def AT(self, y):
        """Apply Aᵀ to y."""
        return self.Amat.T @ y
