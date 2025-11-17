import numpy as np

"""
Linear operators Ax and A^Tx.
Help solvers work with any type of linear transformation — dense, sparse, or implicit
"""

class LinearOperator:
    """linear operator with forward map A(x) and adjoint A^T(y)
	"""

    def __init__(self, forward, adjoint, shape=None):
        """
        forward: callable implementing A(x)
        adjoint: callable implementing A^T(y)
        shape: optional (m, n)
        """
        self.forward = forward
        self.adjoint = adjoint
        self.shape = shape

    def A(self, x):
        """Compute A(x)."""
        return self.forward(x)

    def AT(self, y):
        """Compute A^T(y)."""
        return self.adjoint(y)
