"""
Unit tests for all  implemented proximal operators in src.util.prox_ops.

These tests verify that each prox operator behaves correctly on simple,
deterministic inputs where the expected output can be computed in closed form.
The goal is to ensure numerical correctness and catch implementation bugs early.
"""

import numpy as np
from src.util.prox_ops import (
    prox_quadratic,
    prox_Euclidean_norm,
    prox_l1,
    prox_linf,
    prox_norm2_linear,
    prox_Huber,
    prox_neg_sum_log,
    prox_spectral,
    prox_nuclear,)


def test_prox_l1():
    """test prox_l1"""
    x = np.array([3.0, -2.0, 0.5])
    alpha = 1.0
    expected = np.array([2.0, -1.0, 0.0])
    assert np.allclose(prox_l1(x, alpha), expected)


def test_prox_quadratic():
    """test prox_quadratic"""
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, -1.0])
    x = np.array([3.0, 2.0])
    alpha = 0.5

    I = np.eye(2)
    expected = np.linalg.solve(I + alpha * A, x - alpha * b)
    assert np.allclose(prox_quadratic(x, alpha, A, b), expected)


def test_prox_Euclidean_norm():
    """test prox Euclidean norm shrink"""
    x = np.array([3.0, 4.0])  # norm = 5
    alpha = 2.0
    expected = (1 - alpha / 5) * x
    assert np.allclose(prox_Euclidean_norm(x, alpha), expected)


def test_prox_Euclidean_norm_zero():
    """test prox Euclidean zero-out"""
    x = np.array([0.2, -0.2])
    alpha = 2.0
    assert np.allclose(prox_Euclidean_norm(x, alpha), np.zeros_like(x))


def test_prox_linf():
    """test prox linf shrinks into L1 ball"""
    x = np.array([3.0, 1.0])
    alpha = 1.0
    y = prox_linf(x, alpha)
    diff = x - y
    # Projection residual should lie on the L1 ball boundary
    assert np.isclose(np.sum(np.abs(diff)), alpha)


def test_prox_norm2_linear():
    """test prox ||Ax||_2"""
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    x = np.array([3.0, 4.0])
    alpha = 1.0

    Ax = A @ x
    norm = np.linalg.norm(Ax)
    expected = x - (alpha / norm) * (A.T @ Ax)

    assert np.allclose(prox_norm2_linear(x, alpha, A), expected)


def test_prox_norm2_linear_zero():
    """test prox ||Ax||_2 zero-out"""
    A = np.eye(2)
    x = np.array([0.1, -0.1])
    alpha = 1.0
    assert np.allclose(prox_norm2_linear(x, alpha, A), np.zeros_like(x))


def test_prox_Huber_quadratic():
    """test Huber quadratic region"""
    x = np.array([0.2, -0.3])
    alpha = 0.5
    mu = 1.0
    expected = x / (1 + alpha / mu)
    assert np.allclose(prox_Huber(x, alpha, mu), expected)


def test_prox_Huber_linear():
    """test Huber linear/L1 region"""
    x = np.array([3.0, -4.0])
    alpha = 1.0
    mu = 1.0
    expected = np.array([2.0, -3.0])
    assert np.allclose(prox_Huber(x, alpha, mu), expected)


def test_prox_neg_sum_log():
    """test prox -sum log(x)"""
    x = np.array([1.0, 2.0])
    alpha = 0.5
    expected = (x + np.sqrt(x**2 + 4 * alpha)) / 2
    assert np.allclose(prox_neg_sum_log(x, alpha), expected)


def test_prox_spectral():
    """test spectral prox (top singular shrink)"""
    X = np.array([[3.0, 0.0], [0.0, 1.0]])
    alpha = 1.0

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s[0] = max(s[0] - alpha, 0)
    expected = (U * s) @ Vt

    assert np.allclose(prox_spectral(X, alpha), expected)


def test_prox_nuclear():
    """test nuclear norm prox"""
    X = np.array([[3.0, 0.0], [0.0, 1.0]])
    alpha = 1.0

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - alpha, 0)
    expected = U @ np.diag(s) @ Vt

    assert np.allclose(prox_nuclear(X, alpha), expected)
