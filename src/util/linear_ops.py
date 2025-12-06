import numpy as np

"""
Linear operators Ax and A^T y.
Helps solvers work with any type of linear transformation —
dense matrices, sparse matrices, or implicit operators.
"""

class LinearOperator:
    """
    Linear operator with forward map A(x) and adjoint A^T(y).
    """
    def __init__(self, A, AT=None):
        """
        Parameters:
        A : matrix or callable, forward operator so that A(x) computes Ax.
        AT : callable, optional, adjoint operator so that AT(y) computes A^T y
			if None and A is a matrix, A.T is used.
        """
        self.A = A
        self.AT_op = AT

    def __call__(self, x):
        """
        Apply the forward operator A(x)
        """
        if callable(self.A):
            return self.A(x)
        return self.A @ x

    def T(self, y):
        """
        Apply the adjoint operator A^T(y)
        """
        if self.AT_op is not None:
            return self.AT_op(y)
        return self.A.T @ y
