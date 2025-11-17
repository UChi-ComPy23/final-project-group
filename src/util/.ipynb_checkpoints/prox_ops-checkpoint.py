import numpy as np

"""
Collection of proximal operators.
Each function takes the form prox_f(x, alpha), returning prox_{α f}(x).

Only commonly used operators implemented; the rest are for completeness and future extension.
"""

#common useful operators. WIll implement

def prox_quadratic(x, alpha, A, b):
    """Prox of α(1/2 x^TAx + b^Tx), 𝐀∈𝕊𝑛₊
	"""
    raise NotImplementedError

def prox_Euclidean_norm(x, alpha):
    """Prox of 𝛼⁢∥𝐱∥2
	"""
    raise NotImplementedError

def prox_l1(x, alpha):
    """Prox of α‖x‖₁, 𝑙1−norm
	"""
    raise NotImplementedError

def prox_linf(x, alpha):
    """Prox of α‖x‖_∞, 𝑙∞−norm
	"""
    raise NotImplementedError

def prox_norm2_linear(x, alpha, A):
    """Prox of α‖Ax‖₂, 𝑙2⁢ norm of a linear transformation.
	"""
    raise NotImplementedError

def prox_Huber(x, alpha, mu):
    """Prox of α·Huberₘᵤ(x), μ > 0
	"""
    raise NotImplementedError

def prox_neg_sum_log(x, alpha):
    """Prox of −α Σ log(xᵢ), negative sum of logs.
	"""
    raise NotImplementedError

def prox_spectral(X, alpha):
    """Prox of α‖X‖₂,₂ = ασ₁(X) spectral norm
	"""
    raise NotImplementedError

def prox_nuclear(X, alpha):
    """Prox of α‖X‖_* nuclear norm
	"""
    raise NotImplementedError

# Below will not be implemented for now.

def prox_l1_squared(x, alpha):
    """Prox of α‖x‖₁² squared 𝑙1−norm
	"""
    raise NotImplementedError

def prox_max(x, alpha):
    """Prox of 𝛼⁢max⁡{𝑥1,…,𝑥𝑛}
	"""
    raise NotImplementedError

def prox_sum_k_largest(x, alpha, k):
    """Prox sum of k largest value
	"""
    raise NotImplementedError

def prox_sum_k_largest_abs(x, alpha, k):
    """Prox of sum of k largest absolute values
	"""
    raise NotImplementedError

def prox_max_eigenvalue(X, alpha):
    """Prox of maximum eigenvalue 𝛼⁢𝜆max⁡(𝐗)
	"""
    raise NotImplementedError

def prox_neg_log_det(X, alpha):
    """Prox of −α log(det(X)), X ∈ Sⁿ₊."""
    raise NotImplementedError

