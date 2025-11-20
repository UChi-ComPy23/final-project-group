import numpy as np

"""
Linear operators Ax and A^Tx.
Help solvers work with any type of linear transformation — dense, sparse, or implicit
"""

class LinearOperator:
    """linear operator with forward map A(x) and adjoint A^T(y)
	"""

    def __init__(self, A, AT=None):
        """
        A: matrix or callable
        AT: adjoint (optional when A is callable)
        """
        self.A = A
        self.AT_op = AT

    def __call__(self, x):
        if callable(self.A):
            return self.A(x)
        return self.A @ x

    def T(self, y):
        if self.AT_op is not None:
            return self.AT_op(y)
        return self.A.T @ y