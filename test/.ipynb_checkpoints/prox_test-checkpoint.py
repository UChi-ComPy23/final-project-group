import numpy as np
from src.util.prox_ops import prox_l1

def test_prox_l1_simple():
	"""test if prox_l1 is working as expected
	"""
    x = np.array([3.0, -2, 0.5])
    alpha = 1
    expected = np.array([2.0, -1.0, 0])

    y = prox_l1(x, alpha)
    assert np.allclose(y, expected), "prox_l1 soft-thresholding failed"