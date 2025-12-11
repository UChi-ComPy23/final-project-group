import numpy as np

from src.Problems.lasso import LassoProblem
from src.Problems.constrained import ConstrainedLSProblem
from src.Problems.huber_comp import HuberCompositeProblem
from src.Problems.logistic import LogisticProblem
from src.Problems.quadratic import QuadraticProblem
from src.Problems.quadratic_dual import QuadraticDualProblem
from src.Problems.total_var import TotalVariationProblem
from src.Problems.comirror_box import CoMirrorBoxProblem

from src.Method.prox_gradient import ProxGradient
from src.Method.fista import FISTA
from src.Method.prox_subgradient import ProxSubgradient
from src.Method.sfista import SmoothedFISTA
from src.Method.comd import CoMirrorDescent
from src.Method.adlpmm import ADLPMM
from src.Method.nested_fista import NestedFISTA, accurate_inner_solver
from src.Method.fdpg import FDPG

from src.util.proj import proj_box
from src.util.prox_ops import prox_l1

"""
This test provides correctness and functionality tests for all solvers in the toolbox. Because different methods have
different theoretical guarantees, each solver is tested according to what it is designed to achieve:

• Solvers with convergence guarantees (ProxGradient, FISTA, SmoothedFISTA,
  NestedFISTA, ADLPMM(in special case), FDPG) are tested on tiny synthetic problems where
  the exact minimizer is known. These tests verify that the solver
  converges to the correct solution within tolerance.

• First-order nonsmooth methods without exact minimizer guarantees
  (ProxSubgradient, CoMirrorDescent) are tested differently. Instead of
  checking convergence to an optimum, the tests ensure that objective
  values decrease appropriately or that constraint violations improve,
  and that the methods run without numerical or integration errors.
"""

def test_adlpm_correctness_lasso():
    """a special case that adlpmm produces a exact minimizer, test if ADLPMM solver produces desired minimizer
    """
    A = np.eye(3)
    y = np.array([3.0, -1.0, 0.5])
    lam = 1.0

    problem = LassoProblem(A, y, lam, mode="admm")

    x0 = np.zeros(3)
    z0 = np.zeros(3)
    u0 = np.zeros(3)

    solver = ADLPMM(problem, x0, z0, u0, lam, rho=1.0)
    for _ in range(200):
        solver.step()

    expected = np.array([2, 0, 0]) # soft-threshold
    assert np.allclose(solver.x, expected, atol=1e-2)

def test_proxgradient_correctness_lasso():
    """ProxGradient with composite BTLS should converge to the Lasso minimizer (A = I case)."""
    A = np.eye(3)
    y = np.array([3.0, -1, 0.5])
    lam = 1.0

    problem = LassoProblem(A, y, lam)
    x0 = np.zeros(3)

    solver = ProxGradient(problem, x0, alpha=None, btls=True)
    for _ in range(200):
        solver.step()

    x_pg = solver.x
    # exact soft-thresholding solution for A = I
    expected = np.array([2, 0, 0])

    assert np.allclose(x_pg, expected, atol=1e-2)

def test_fista_correctness_lasso():
    """FISTA with composite BTLS should converge faster to the Lasso minimizer (A = I case)."""
    A = np.eye(3)
    y = np.array([3.0, -1, 0.5])
    lam = 1.0

    problem = LassoProblem(A, y, lam)
    x0 = np.zeros(3)

    solver = FISTA(problem, x0, alpha=None)
    for _ in range(100):
        solver.step()

    x_fista = solver.x
    expected = np.array([2.0, 0, 0])

    assert np.allclose(x_fista, expected, atol=5e-3)


def test_smoothed_fista_correctness_quadratic_dual():
    """
    SmoothedFISTA should converge to x*=0 for the problem
        0.5 ||x||^2 + lam ||x||_1.
    """
    np.random.seed(0)

    lam = 0.1
    problem = QuadraticDualProblem(lam)
    x0 = np.random.randn(5)

    # Choose smoothing parameter and Lipschitz constant
    mu = 1e-2
    L = 1 + (1 / mu) # safe bound for ∇F_μ

    solver = SmoothedFISTA(problem, x0, mu, L)
    for _ in range(300):
        solver.step()

    x_final = solver.x
    x_star = np.zeros_like(x_final)

    # Smoothed FISTA should converge very close to 0
    assert np.allclose(x_final, x_star, atol=1e-2)


def test_nested_fista_correctness_huber():
    """
    NestedFISTA should reliably decrease the composite objective on a small
    nonlinear-composite Huber + L1 problem: F(x) = phi(f(x)) + lam * g(Ax)
    Because the inner solver is inexact (ADMM), we compare only against a 
    high-accuracy ProxGradient baseline in *objective value*, not iterates.
    """
    np.random.seed(0)

    # small synthetic problem
    m, n = 10, 5
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    delta = 0.1
    lam = 0.1

    # composite problem (must match your class name exactly)
    problem = HuberCompositeProblem(A=A,b=b,delta_f=delta,lam=lam, phi_mode="sqrt")

    x0 = np.zeros(n)
    nf = NestedFISTA(problem, x0, lam, accurate_inner_solver)
    for _ in range(300):
        nf.step()

    x_nf = nf.x
    pg = ProxGradient(problem, np.zeros(n), alpha=1e-3)
    for _ in range(5000):
        pg.step()
    x_star = pg.x

    # final objective (must use problem.phi + problem.f + lam*g(Ax))
    def F(x):
        return problem.phi(problem.f(x)) + lam * problem.g(problem.A(x))
		
    assert abs(F(x_nf) - F(x_star)) < 5e-2
    assert np.linalg.norm(x_nf - x_star) < 5e-1


def test_fdpg_correctness_quadratic():
    """
    FDPG correctness test on a tiny quadratic problem:
        f(x) = 0.5 ||x||^2
        g(z) = lam * ||z||_1
        A = identity
    Exact minimizer is x* = 0 for any lam > 0.
    """
    lam = 0.5
    problem = QuadraticDualProblem(lam)

    y0 = np.array([1, -2, 3]) # initial dual variable (arbitrary)
    solver = FDPG(problem, x0=y0, L=1)  
    # L = normA^2 / mu = 1 / 1 = 1

    # run FDPG iterations
    for _ in range(200):
        solver.step()

    # recover primal x(y) = grad f*( -A^T y ) = -(y)
    x_fdpg = solver.history["x_primal"][-1]

    expected = np.zeros_like(x_fdpg)

    # must converge to zero vector
    assert np.allclose(x_fdpg, expected, atol=1e-3)

def test_comd_runs_and_respects_constraints():
    """
    CoMirrorDescent is meant for constrained convex problems.
    A correct test is:

    - solver runs without error,
    - objective decreases over iterations,
    - constraints are satisfied (x stays in [l,u]),
    - iterates remain finite.

    We use the CoMirrorBoxProblem, which matches COMD’s oracle requirements.
    """
    np.random.seed(0)

    m, n = 30, 5
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # box constraints
    l = np.zeros(n)          # x >= 0
    u = np.ones(n) * 2.0     # x <= 2

    problem = CoMirrorBoxProblem(A, b, l, u)
    x0 = np.ones(n) # feasible starting point

    solver = CoMirrorDescent(problem, x0, step_size=1e-1)
    for _ in range(200):
        solver.step()

    x_final = solver.x

    # objective must decrease
    f0 = problem.f(x0)
    fT = problem.f(x_final)
    assert fT <= f0 + 1e-8

    # feasibility: x ∈ [l, u]
    assert np.all(x_final >= l - 1e-8)
    assert np.all(x_final <= u + 1e-8)

    # iterates finite
    assert np.all(np.isfinite(x_final))


def test_prox_subgradient_runs_and_decreases_objective():
    """
    ProxSubgradient does not yield exact minimizers, so a correct test checks:
    - the solver executes without errors,
    - the objective decreases sufficiently over iterations,
    - iterates remain finite.

    We use a simple LASSO objective f(x) + lam‖x‖₁ with prox_g = soft-thresholding.
    """
    np.random.seed(0)

    m, n = 30, 5
    A = np.random.randn(m, n)
    y = np.random.randn(m)
    lam = 0.1

    problem = LassoProblem(A, y, lam)
    x0 = np.zeros(n)
    solver = ProxSubgradient(problem, x0, step_rule=lambda k: 1e-2 / np.sqrt(k + 1))

    for _ in range(300):
        solver.step()

    x_final = solver.x

    # objective at start and end
    f0 = problem.f(x0) + problem.g(x0)
    fT = problem.f(x_final) + problem.g(x_final)

    # must decrease (subgradient is noisy → allow tolerance)
    assert fT <= f0 + 1e-3

    # iterates finite
    assert np.all(np.isfinite(x_final))

