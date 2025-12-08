import numpy as np

"""
Unit tests for all implemented projection operators in src.util.proj.

Each projection is tested using simple cases where the closed-form
projection outcome is known exactly. These tests verify correctness
and catch implementation bugs early.
"""
from src.util.proj import (
    proj_Euclidean_ball,
    proj_box,
    proj_affine_set,
    proj_halfspace,
    proj_Lorentz,
    proj_simplex,
    proj_l1_ball,
    proj_psd,
    proj_spectral_ball,
    proj_nuclear_ball,)


def test_proj_Euclidean_ball_inside():
    """Point inside ball should remain unchanged."""
    x = np.array([1, 1])
    c = np.array([0, 0])
    r = 3.0
    assert np.allclose(proj_Euclidean_ball(x, c, r), x)


def test_proj_Euclidean_ball_outside():
    """Projection scales toward center."""
    x = np.array([4, 0])
    c = np.array([0, 0])
    r = 2
    expected = np.array([2, 0])
    assert np.allclose(proj_Euclidean_ball(x, c, r), expected)


def test_proj_box():
    """Box projection clamps coordinates into [l, u]."""
    x = np.array([3, -2, 0.5])
    l = np.array([0, 0, 0])
    u = np.array([2, 1, 1])
    expected = np.array([2, 0, 0.5])
    assert np.allclose(proj_box(x, l, u), expected)


def test_proj_affine_set():
    """Projection onto { Ax = b } via closed form."""
    A = np.array([[1, 1]])
    b = np.array([1])
    x = np.array([2, 2])

    # Closed form: projection onto line x1 + x2 = 1
    # Solve min ||x - z||^2 s.t. z1 + z2 = 1
    # The projection is midpoint - shift*normal
    expected = np.array([0.5, 0.5])
    assert np.allclose(proj_affine_set(x, A, b), expected)


def test_proj_halfspace_inside():
    """Point inside half-space unchanged."""
    x = np.array([1, 1])
    a = np.array([1, 1])
    b = 3
    assert np.allclose(proj_halfspace(x, a, b), x)


def test_proj_halfspace_outside():
    """Projection onto boundary <a,z> = b."""
    x = np.array([3, 3])
    a = np.array([1, 1])
    b = 3
    # Closed form: project orthogonally onto x1 + x2 = 3
    expected = np.array([1.5, 1.5])
    assert np.allclose(proj_halfspace(x, a, b), expected)


def test_proj_Lorentz_inside():
    """Point inside Lorentz cone unchanged."""
    x = np.array([0.5, 0.5, 2])  # ||v|| = sqrt(0.5^2+0.5^2)=0.707 <= 2
    assert np.allclose(proj_Lorentz(x), x)


def test_proj_Lorentz_outside():
    """Projection onto Lorentz cone boundary."""
    x = np.array([3, 0, 1])  # ||v|| = 3 > t = 1
    # formula: (v,t) maps to ((||v||+t)/(2||v||)) v , (||v||+t)/2
    nv = 3
    t = 1
    coef = (nv + t) / (2 * nv)
    expected = np.array([coef * 3, 0, (nv + t) / 2])
    assert np.allclose(proj_Lorentz(x), expected)


def test_proj_simplex():
    """Projection onto simplex sum(x)=1, x>=0."""
    x = np.array([0.5, 2, -1])
    expected = np.array([0, 1, 0])  # manually known
    out = proj_simplex(x, r=1)
    assert np.allclose(out.sum(), 1)
    assert np.all(out >= 0)
    assert np.allclose(out, expected)


def test_proj_l1_ball_inside():
    """If already inside L1 ball, unchanged."""
    x = np.array([0.2, -0.3])
    assert np.allclose(proj_l1_ball(x, r=1), x)


def test_proj_l1_ball_outside():
    """Projection shrinks onto L1-ball boundary."""
    x = np.array([3, -1])
    r = 2
    y = proj_l1_ball(x, r)
    assert np.isclose(np.sum(np.abs(y)), r)


def test_proj_psd():
    """PSD projection clips negative eigenvalues."""
    X = np.array([[2, 0], [0, -1]])
    out = proj_psd(X)
    eigvals = np.linalg.eigvalsh(out)
    assert np.all(eigvals >= -1e-10)


def test_proj_spectral_ball():
    """Projection caps largest singular value to r."""
    X = np.array([[3, 0], [0, 1]])
    r = 2
    Y = proj_spectral_ball(X, r)
    s = np.linalg.svd(Y, compute_uv=False)
    assert s[0] <= r + 1e-10


def test_proj_nuclear_ball():
    """Projection ensures nuclear norm <= r."""
    X = np.array([[3, 0], [0, 1]])
    r = 2.5
    Y = proj_nuclear_ball(X, r)
    s = np.linalg.svd(Y, compute_uv=False)
    assert np.isclose(np.sum(s), r)
