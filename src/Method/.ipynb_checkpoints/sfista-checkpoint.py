import numpy as np
from core.solver import SolverBase

"""
smoothed FISTA
Minimization model: minimize f(x) + λ_g g(Ax) + λ_h h(x)

Assumptions:
-f smooth
-g is proper, closed, proximable
-h is proper, closed, proximable
-λ_g > 0, λ_h > 0

Oracles:
- f(x) & ∇f(x)
- g(x) & prox_{αg}(x)
- h(x) & prox_{αh}(x)
- A(x) & A^T(y)
"""

class SmoothedFISTA(SolverBase):
    """Smoothed FISTA method
    """
    def __init__(self, problem, x0):
        """
        problem: ProblemBase 
        x0: initial point.
        """
        super().__init__(problem, x0)

    def step(self):
        """Perform one smoothed FISTA update
        """
        raise NotImplementedError


# Restart-FISTA（S-FISTA）
def restart_fista(f, grad_f, prox_g, x0, L, max_iter=500):
    """
    Uses standard FISTA updates, automatically resets the momentum parameter t when the
    update direction is not aligned with previous step.
    """
    x = x0.copy()
    y = x.copy()
    t = 1.0
    objs = []
    for k in range(max_iter):
        x_new = prox_g(y - (1/L) * grad_f(y), 1.0/L)

        if np.dot(x_new - x, y - x_new) > 0:
            # restart
            y = x_new
            t = 1
        else:
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = x_new + (t - 1) / t_new * (x_new - x)
            t = t_new

        x = x_new
        objs.append(f(x))

    return x, np.array(objs)

