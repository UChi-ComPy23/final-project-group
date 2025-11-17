"""
The base class for defining optimization problems, i.e. oracle interface.
The solvers require various oracles as inputs, each oracle is a python function handle. 
For a function f and a linear operator A, the following oracle notations are used:

- function value of f

- a subgradient of f (𝐱↦𝑓'(𝐱)∈∂𝑓⁡(𝐱))

- gradient of f (𝐱↦∇𝑓⁡(𝐱))

- gradient of the conjugate of f (𝐱↦argmax⁡{⟨𝐮,𝐱⟩−𝑓⁡(𝐮)})

- proximal operator of a positive constant times the function ((𝐱,𝛼)↦prox𝛼⁢𝑓⁢(𝐱))

- linear transformation A (𝐱↦A⁢𝐱)

- adjoint of A (𝐱↦A^T⁢𝐱)

All the involved functions are convex.
"""

class ProblemBase:
    """Base class for defining optimization problems.
    """
	
	def f(self, x):
        """Return the function value f(x)
		"""
        raise NotImplementedError

    def subgrad(self, x):
        """Return a subgradient f'(x) ∈ ∂f(x)
		"""
        raise NotImplementedError

    def grad(self, x):
        """Return the gradient ∇f(x) of f at x. f is smooth.
		"""
        raise NotImplementedError

    def grad_conjugate(self, x):
        """Return gradient of the conjugate function ∇f^*(x)
		"""
        raise NotImplementedError

    def prox_f(self, x, alpha):
        """
        Return the proximal operator of α f:
            prox_{α f}(x) = argmin_u ( α f(u) + 1/2 ||u - x||^2 )
        """
        raise NotImplementedError

    def A(self, x):
        """Return the linear transformation A(x)
		"""
        raise NotImplementedError

    def AT(self, y):
        """Return the adjoint linear transformation A^T(y)
		"""
        raise NotImplementedError

