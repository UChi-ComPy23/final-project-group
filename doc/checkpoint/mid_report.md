# 1. Introduction

Convex optimization, as a fundamental class of optimization problems, has widespread applications in many areas of applied mathematics and engineering. A typical convex problem minimizes/maximizes a real-valued, convex, coercive, and \(L\)-smooth function \(f\) defined on a closed, convex set \(E\). When \(f\) is differentiable, it is automatically proper and lower semicontinuous. Under these conditions, it is guaranteed that the minimization problem admits at least one solution, and strict convexity further ensures uniqueness (Dossal, Hurault, Papadakis, 2024).

Because of these structural properties, first-order methods, which rely only on gradient or subgradient information, are often sufficient to solve such problems efficiently. Compared to second-order methods, first-order algorithms are significantly more scalable and computationally affordable, making them well suited for large-scale convex optimization.

In this project, we develop a Python toolbox implementing a collection of classical first-order algorithms for convex problems, inspired by the MATLAB package FOM developed by Beck and Guttmann-Beck (2018). Our implementation follows the theoretical framework summarized in Optimization with First-Order Algorithms (Dossal, Hurault & Papadakis, 2024).

The general models we target include composite formulations such as \[\min_x\; f(x) + \lambda\, g(Ax),  \qquad \min_x\; f(x)\quad \text{s.t.}\quad g_i(x) \le 0,\]
where \(f\), \(g\), and each \(g_i\) are convex, and \(A\) is a linear map. These structures encompass common applications in constrained optimization and inverse problems.

# 2. Numerical Methods and Notation

This project implements eight first-order optimization algorithms, following the models and assumptions in Beck and Guttmann-Beck (2018) and the theoretical framework of Dossal, Hurault & Papadakis (2024). These are first-order methods in the sense that they rely solely on gradients, subgradients, linear maps, and proximal operators.

The solvers included in our toolbox are:
- proximal subgradient  
- co-mirror descent (COMD)  
- proximal gradient   
- FISTA  
- smoothed FISTA (SFISTA)  
- alternating direction linearized proximal multiplier method (ADLPMM)  
- nested FISTA  
- fast dual proximal gradient (FDPG)
Each solver corresponds to a standard composite convex model such as  
\[\min_x\; f(x) + \lambda g(x), \qquad\min_x\; f(x) + \lambda g(Ax), \qquad\min_x\; f(x)\;\text{s.t. } g_i(x)\le 0,\]
with smoothness, Lipschitz continuity, or proximability assumptions depending on the method.

## Oracle interface

All solvers use a unified oracle interface:
- \(f(x)\): function value  
- \(f'(x)\) or \(\nabla f(x)\): subgradient or gradient  
- proximal operator  \[\mathrm{prox}_{\alpha g}(x)= \arg\min_u\left(g(u) + \frac{1}{2\alpha}\|u-x\|^2\right)\]
- linear map \(A(x)\) and adjoint \(A^T(y)\)

## Iteration rules

Each solver implements a specific update rule. 

- Proximal gradient: \[x^{k+1}= \mathrm{prox}_{\alpha g}\left(x^k - \alpha \nabla f(x^k)\right)\]

- Co-mirror descent: Handles constrained problems over simple domains using mirror maps.

- FISTA:  Applies Nesterov acceleration on top of proximal gradient.

- Smoothed FISTA (SFISTA): Uses a smooth approximation of \(g(Ax)\).

- ADLPMM: Alternates an x-update, a z-update via \(\mathrm{prox}_{g}\), and a dual update.

- FDPG: Uses proximal gradient applied to the dual and requires access to \(\nabla f^*\).

## Step size selection

When no fixed step size is provided, smooth solvers use backtracking line search (BTLS) with the Armijo condition:
\[f(x + \alpha p) \le f(x) + c\,\alpha\, \nabla f(x)^T p\]
Nonsmooth components rely on closed-form proximal operators or projections (such as \(\ell_1\), \(\ell_2\), nuclear norm, spectral norm, or simple box and ball constraints).

## Computational tools

Since the computation involved in first-order methods is relatively simple, our implementation relies only on standard scientific Python libraries. In particular, we use NumPy and SciPy for linear algebra, sparse matrices and linear operators.

# 3. `src` structure

The `src` directory has three main subpackages: `core`, `methods`, and `utils`. The `core` folder contains the base classes for the problem definition and the solver, including the oracle interface and the main run loop. The `methods` folder includes all the first-order algorithms described in Section 2, each written as a subclass of the solver base class with its own update rule. The `utils` folder provides helper functions such as proximal operators, projections, backtracking line search, and linear operator wrappers for both dense and sparse matrices. The original project also includes a `tools.py` file with simple helper functions that do not directly involve the algorithms, such as plotting convergence curves. 

# References

Beck, A., & Guttmann-Beck, N. (2018). FOM – a MATLAB toolbox of first-order methods for solving convex optimization problems. Optimization Methods and Software, 34(1), 172–193. https://doi.org/10.1080/10556788.2018.1437159  

Dossal, C., Hurault, S., & Papadakis, N. (2024). Optimization with first order algorithms. arXiv. https://doi.org/10.48550/arXiv.2410.19506