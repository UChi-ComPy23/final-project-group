import numpy as np

"""
Orthogonal projection functions onto convex sets.
Only commonly used functions implemented; the rest are for completeness and future extension.
"""

# Projection onto Euclidean ball
def proj_Euclidean_ball(x, c, r):
    """
    Projection onto Euclidean ball B[c, r] = { x : ||x - c|| <= r }.
    """
    diff = x - c
    norm = np.linalg.norm(diff)
    if norm <= r:
        return x
    return c + diff * (r / norm)

# Projection onto a box
def proj_box(x, l, u):
    """Projection onto a coordinate-wise box {x : l <= x <= u}."""
    return np.minimum(np.maximum(x, l), u)

# Projection onto affine set
def proj_affine_set(x, A, b):
    """
    Projection onto affine set {x : A x = b}, A full row rank.

        P(x) = x - Aᵀ (A Aᵀ)^{-1} (A x - b)
    """
    At = A.T
    return x - At @ np.linalg.solve(A @ At, A @ x - b)

# Projection onto half-space
def proj_halfspace(x, a, b):
    """
    Projection onto half-space H^-(a, b) = { x : <a, x> <= b }.
    If x already satisfies constraint, projection = x.
    Otherwise project onto boundary hyperplane <a, z> = b.
    """
    ax = np.dot(a, x)
    if ax <= b:
        return x
    return x - (ax - b) / (np.dot(a, a) + 1e-12) * a

# Projection onto Lorentz cone
def proj_Lorentz(x):
    """
    Projection onto Lorentz cone L_n = { (v, t) : ||v|| <= t }.

    Input x = [v; t].
    """
    v = x[:-1]
    t = x[-1]
    nv = np.linalg.norm(v)

    # inside cone
    if nv <= t:
        return x.copy()

    # opposite cone: project to 0
    if nv <= -t:
        return np.zeros_like(x)

    # general case
    coef = (nv + t) / (2 * nv + 1e-12)
    v_proj = coef * v
    t_proj = (nv + t) / 2
    out = np.zeros_like(x)
    out[:-1] = v_proj
    out[-1] = t_proj
    return out

# Projection onto simplex
def proj_simplex(x, r=1.0):
    """
    Projection onto simplex Δ(r) = { x : sum x = r, x >= 0 }.

    Standard algorithm: sorting + thresholding.
    """
    v = x.copy()
    n = len(v)
    u = np.sort(v)[::-1]           # descending
    cssv = np.cumsum(u)
    rho = np.where(u + (r - cssv) / (np.arange(n) + 1) > 0)[0][-1]
    theta = (cssv[rho] - r) / (rho + 1)
    return np.maximum(v - theta, 0)

# Projection onto L1 ball
def proj_l1_ball(x, r):
    """
    Projection onto ℓ1-ball { x : ||x||_1 <= r }.

    Same formula used for l-infinity proximal via Moreau.
    """
    if np.linalg.norm(x, 1) <= r:
        return x.copy()

    u = np.abs(x)
    w = -np.sort(-u)               # descending
    cssv = np.cumsum(w)
    rho = np.where(w > (cssv - r) / (np.arange(len(w)) + 1))[0][-1]
    theta = (cssv[rho] - r) / (rho + 1)
    return np.sign(x) * np.maximum(u - theta, 0)

# Projection onto PSD cone
def proj_psd(X):
    """
    Projection onto Positive Semi-Definite cone S_+.

        P(X) = Q diag(max(λ_i, 0)) Qᵀ
    """
    Xsym = 0.5 * (X + X.T)
    eigvals, eigvecs = np.linalg.eigh(Xsym)
    eigvals_clipped = np.maximum(eigvals, 0)
    return eigvecs @ (eigvals_clipped[:, None] * eigvecs.T)

# Projection onto spectral-norm ball
def proj_spectral_ball(X, r):
    """
    Projection onto spectral-norm ball { X : σ₁(X) <= r }.

    Clamp singular values: s_i' = min(s_i, r)
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_clipped = np.minimum(s, r)
    return (U * s_clipped) @ Vt

# Projection onto nuclear-norm ball
def proj_nuclear_ball(X, r):
    """
    Projection onto nuclear-norm ball { X : sum(s_i) <= r }.

    Equivalent to projecting singular values onto L1-ball of radius r.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    if s.sum() <= r:
        return X.copy()

    # project s onto ℓ₁ ball
    w = -np.sort(-s)
    cssv = np.cumsum(w)
    rho = np.where(w > (cssv - r) / (np.arange(len(w)) + 1))[0][-1]
    theta = (cssv[rho] - r) / (rho + 1)
    s_proj = np.maximum(s - theta, 0)

    return (U * s_proj) @ Vt



# Below will not be implemented for now.

def proj_two_halfspaces(x, a1, b1, a2, b2):
    """Projection onto intersection of two half-spaces
    """
    raise NotImplementedError

def proj_hyperplane_box(x, a, b, l, u):
    """Projection onto intersection of hyperplane and box
    """
    raise NotImplementedError

def proj_halfspace_box(x, a, b, l, u):
    """Projection onto intersection of half-space and box
    """
    raise NotImplementedError

def proj_product(x, r):
    """Projection onto product-superlevel set
    """
    raise NotImplementedError

def proj_l1ball_box(x, w, r, u):
    """Projection onto intersection of weighted ℓ₁-ball and box
    """
    raise NotImplementedError

def proj_spectral_box_sym(X, l, u):
    """Projection onto symmetric spectral box
    """
    raise NotImplementedError

def proj_spectahedron(X, r):
    """Projection onto r-spectahedron (or full spectahedron)
    """
    raise NotImplementedError
