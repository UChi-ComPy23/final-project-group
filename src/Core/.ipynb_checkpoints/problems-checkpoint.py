"""
The base class for defining optimization problems, i.e. oracle interface.
The solvers require various oracles as inputs, each oracle is a python function handle. 
For a function f and a linear operator A, the following oracle notations are used:

- function value of f
- a subgradient of f (x ↦ f'(x) ∈ ∂f(x))
- gradient of f (x ↦ ∇f(x))
- gradient of the conjugate of f (x ↦ argmax_u {⟨u, x⟩ − f(u)})
- proximal operator of a positive constant times the function ((x, α) ↦ prox_{α f}(x))
- linear transformation A (x ↦ A x)
- adjoint of A (x ↦ A^T x)

All the involved functions are convex.
"""

class ProblemBase:
    """Base class for defining optimization problems.
    """

    def f(self, x):
        """
        Return the function value f(x)
        """
        raise NotImplementedError

    def subgrad(self, x):
        """
        Return a subgradient f'(x) ∈ ∂f(x)
        """
        raise NotImplementedError

    def grad(self, x):
        """
        Return the gradient ∇f(x) of f at x. f is smooth.
        """
        raise NotImplementedError

    def grad_conjugate(self, x):
        """
        Return gradient of the conjugate function ∇f*(x)
        """
        raise NotImplementedError

    def prox_f(self, x, alpha):
        """
        Return the proximal operator of α f:
            prox_{α f}(x) = argmin_u ( α f(u) + 1/2 ||u - x||^2 )
        """
        raise NotImplementedError

    def prox_g(self, x, alpha):
        """
        Proximal operator of g
        """
        raise NotImplementedError

    def A(self, x):
        """
        not all problems need this. optional attributes, default set to identity.
        """
        return x

    def AT(self, y):
        """
        not all problems need this. optional attributes, default set to identity.
        """
        return y

