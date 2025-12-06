import numpy as np

"""
Collection of proximal operators.
Each function takes the form prox_f(x, alpha), returning prox_{α f}(x).

Only commonly used operators implemented; the rest are for completeness and future extension.
"""

import numpy as np

def prox_quadratic(x, alpha, A, b):
    """
    prox_{α(1/2 x^T A x + b^T x)}(x0) = (I + αA)^(-1)(x0 - αb)
    Closed form because A SPD.
    """
    I = np.eye(A.shape[0])
    return np.linalg.solve(I + alpha*A, x - alpha*b)


def prox_Euclidean_norm(x, alpha):
    """
    prox_{α||x||2}(x0) = (1 - α/||x||)_+ x0
    Shrinks entire vector, not component-wise.
    """
    norm = np.linalg.norm(x)
    if norm <= alpha:
        return np.zeros_like(x)
    return (1 - alpha/norm) * x


def prox_l1(x, alpha):
    return np.sign(x)*np.maximum(np.abs(x)-alpha, 0)


def prox_linf(x, alpha):
    """
    prox_{α||x||∞}(x0) = x0 - proj_{L1 ball radius α}(x0)
    """

    # Projection onto L1 ball
    a = np.abs(x)
    if a.sum() <= alpha:   # already inside L1 ball
        return np.zeros_like(x)

    # soft-threshold via sorting
    w = np.sort(a)[::-1]
    cs = np.cumsum(w)
    rho = np.where(w > (cs - alpha) / (np.arange(len(w)) + 1))[0][-1]
    theta = (cs[rho] - alpha) / (rho + 1)

    return x - np.sign(x)*np.maximum(a - theta, 0)


def prox_norm2_linear(x, alpha, A):
    """
    prox_{α||Ax||2}(x0) = x0 - α*A^T*(Ax0)/max(||Ax0||, α)
    (More general form uses Moreau decomposition)
    """
    Ax = A.dot(x)
    norm = np.linalg.norm(Ax)

    if norm <= alpha:
        return np.zeros_like(x)

    return x - (alpha/norm) * A.T.dot(Ax)


def prox_Huber(x, alpha, mu):
    """
    Huber penalty:
        H(x)= x^2/(2μ) if |x|≤μ,
        H(x)=|x|-μ/2 otherwise.
    Prox applied element-wise.
    """
    out = np.zeros_like(x)
    absx = np.abs(x)

    # region 1: quadratic
    idx1 = absx <= mu + alpha
    out[idx1] = x[idx1] / (1 + alpha/mu)

    # region 2: L1 shrink
    idx2 = absx > mu + alpha
    out[idx2] = np.sign(x[idx2]) * (absx[idx2] - alpha)

    return out


def prox_neg_sum_log(x, alpha):
    """
    Solve separately for each component:
        min_y  -αlog(y)+1/(2)||y-x||^2
        → quadratic equation: y^2 - xy - α = 0
    y = (x + sqrt(x^2 + 4α))/2    with y>0
    """
    return (x + np.sqrt(x**2 + 4*alpha)) / 2


def prox_spectral(X, alpha):
    """
    prox = X - α*u1*v1^T, where u1,v1 are leading singular vectors.
    Requires top SVD (economy version)
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s[0] = max(s[0] - alpha, 0)
    return (U * s) @ Vt


def prox_nuclear(X, alpha):
    """
    prox_{α||X||_*}(X) = U*soft(s,α)*V^T
    shrink singular values elementwise
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - alpha, 0)
    return U @ np.diag(s) @ Vt


# Below will not be implemented for now.

def prox_l1_squared(x, alpha):
    """
	Prox of α‖x‖₁² squared 𝑙1−norm
	"""
    raise NotImplementedError

def prox_max(x, alpha):
    """
	Prox of 𝛼⁢max⁡{𝑥1,…,𝑥𝑛}
	"""
    raise NotImplementedError

def prox_sum_k_largest(x, alpha, k):
    """
	Prox sum of k largest value
	"""
    raise NotImplementedError

def prox_sum_k_largest_abs(x, alpha, k):
    """
	Prox of sum of k largest absolute values
	"""
    raise NotImplementedError

def prox_max_eigenvalue(X, alpha):
    """
	Prox of maximum eigenvalue 𝛼⁢𝜆max⁡(𝐗)
	"""
    raise NotImplementedError

def prox_neg_log_det(X, alpha):
    """
	Prox of −α log(det(X)), X ∈ Sⁿ₊.
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

