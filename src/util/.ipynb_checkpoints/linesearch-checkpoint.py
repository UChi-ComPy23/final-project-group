"""
Line search method used by solvers.
"""

def backtracking(x, p, g, f, alpha0=1.0, rho=0.5, c_ls=1e-4):
    """
    Backtracking line search (BTLS). Return step size alpha satisfy Armijo condition.
	special parameters: 
	p: search direction
	g: gradient at x
	alpha0: initial step size.
    rho: contraction factor (0<rho<1).
    c_ls: Armijo constant.
	"""

# backtracking for 348 Applied Optimization for reference
# def backtracking(x, p, g, alpha0=1.0, rho=0.5, c_ls=1e-4):
#     """find the step size using BTLS
#     """
#     alpha = alpha0
#     fx = f(x)
#     while not in_domain(x + alpha * p): # check if in domain
#         alpha *= rho
#         if alpha < 1e-16:
#             raise RuntimeError("btls failed find step")
#     while f(x + alpha * p) > fx + c_ls * alpha * (g @ p): # Armijo not satisfied
#         alpha *= rho
#         if alpha < 1e-16:
#             raise RuntimeError("btls failed Armijo condition")
#         while not in_domain(x + alpha * p): # maintain feasibility
#             alpha *= rho
#             if alpha < 1e-16:
#                 raise RuntimeError("btls lost feasibility")
#     return alpha


def fixed_step(alpha):
    """Return a fixed constant step size
	"""
    return alpha