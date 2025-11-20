import numpy as np
from src.util.proj import proj_box
from src.util.linesearch import backtracking
from src.util.linear_ops import LinearOperator

def test_proj_box():
	"""test proj_box is working properly.
	"""
    x = np.array([5, -3, 1])
    l = np.array([0, -1, 0])
    u = np.array([3, 2, 1])
    
    expected = np.array([3, -1, 1]) #each coordinate cut to [l, u]
    y = proj_box(x, l, u)
    assert np.allclose(y, expected), "proj_box failed"