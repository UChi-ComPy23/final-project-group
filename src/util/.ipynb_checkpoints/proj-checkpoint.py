import numpy as np

"""
Orthogonal projection functions onto convex sets.
Only commonly used functions implemented; the rest are for completeness and future extension.
"""

#common useful operators. WIll implement

def proj_Euclidean_ball(x, c, r):
    """Projection onto Euclidean ball 𝐵⁡[𝐜,𝑟]={𝐱:∥𝐱−𝐜∥≤𝑟}
    """
    raise NotImplementedError

def proj_box(x, l, u):
    """Projection onto a coordinate-wise box Box⁡[𝐥,𝐮]={𝐱:𝐥≤𝐱≤𝐮}
    """
    return np.minimum(np.maximum(x, l), u)

def proj_affine_set(x, A, b):
    """Projection onto an affine set {𝐱:𝐀𝐱=𝐛}. A full row rank.
    """
    raise NotImplementedError

def proj_halfspace(x, a, b):
    """Projection onto half-space H⁻(a, b) = { x : <a, x> <= b }.
    """
    raise NotImplementedError

def proj_Lorentz(x):
    """Projection onto Lorentz cone 𝐿𝑛={𝐱∈ℝ𝑛+1:∥𝐱{1,…,𝑛}∥≤𝑥𝑛+1} 
    """
    raise NotImplementedError

def proj_simplex(x, r=1.0):
    """Projection onto simplex Δ𝑛⁡(𝑟)={𝐱:𝐞T⁢𝐱=𝑟,𝐱≥𝟎}
    Also full simplex: eᵀx <= r.
    """
    raise NotImplementedError

def proj_l1_ball(x, r):
    """Projection onto ℓ₁-ball {𝐱:∥𝐱∥1≤𝑟}
    """
    raise NotImplementedError

def proj_psd(X):
    """Projection onto Positive Semi definite cone.
    """
    raise NotImplementedError

def proj_spectral_ball(X, r):
    """Projection onto spectral-norm ball. 𝐵∥⋅∥𝑆∞⁡[𝟎,𝑟]={𝐗:𝜎1⁡(𝐗)≤𝑟}
    """
    raise NotImplementedError

def proj_nuclear_ball(X, r):
    """Projection onto nuclear-norm ball:
       { X : sum σᵢ(X) <= r }.
    """
    raise NotImplementedError



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
