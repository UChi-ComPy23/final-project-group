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

The `src` directory has four main subpackages: `core`, `methods`, `utils`, and `problems`. The `core` folder contains the base classes for the problem definition and the solver, including the oracle interface and the main run loop. The `methods` folder includes all the first-order algorithms described in Section 2, each implemented as a subclass of the solver base class with its own update rule. The `utils` folder provides helper functions such as proximal operators, projections, backtracking line search, and linear operator wrappers for both dense and sparse matrices, along with the original `tools.py` file for plotting and other small utilities. The `problems` folder contains concrete optimization problems, such as quadratic, LASSO, logistic regression, and constrained models, each inheriting from the base oracle and supplying the specific function values, gradients, proximal operators, and linear mappings required by the solvers.

# 4. Description of Test

We have constructed a comprehensive Lasso sparse regression testing framework to systematically compare the performance of three optimization algorithms: the Proximal Gradient (PG) method, the FISTA accelerated method, and the Smoothed FISTA (S-FISTA). The tests are based on a $200 \times 1000$-dimensional sparse signal recovery problem, using a unified experimental setup: a fixed regularization parameter $\lambda = 0.1$, 800 iterations, and a step size set to the theoretically optimal value of $1/L$.

PG employs the classical forward-backward splitting mechanism, achieving optimization through alternating gradient steps and proximal operator applications. FISTA introduces Nesterov momentum acceleration on top of PG, constructing extrapolation points using historical gradient information. S-FISTA further incorporates an intelligent restart mechanism based on gradient conditions, specifically designed to handle complex optimization problems with multiple composite terms. The testing framework includes complete convergence trajectory recording, computation time statistics, and visualization tools, ensuring the comprehensiveness and reliability of the comparative experiments.

Here is the result of the test:

![PG vs FISTA Convergence Curve Comparison](../../Figur_1.png)

The convergence curve comparison chart clearly demonstrates the performance differences between the Proximal Gradient (PG) method and the FISTA accelerated method on the Lasso problem. From the chart, it can be observed that FISTA (red line) exhibits significant convergence acceleration, rapidly reducing the gap between the objective function value and the optimal value to approximately 10^-5 within the first 100 iterations, reflecting its theoretical O(1/k²) superlinear convergence characteristic. In contrast, PG (blue line) shows a typical linear convergence pattern, with a smoother and more stable convergence trajectory. Notably, FISTA exhibits slight oscillations during the convergence process, which is a typical phenomenon of the Nesterov momentum method, while PG maintains a perfect monotonic descent characteristic. After 800 iterations, both methods achieve high precision above 10^-13, with FISTA attaining a better final convergence accuracy. This comparison not only validates the correctness of both algorithm implementations but also provides clear guidance for algorithm selection in practical applications: FISTA should be chosen when pursuing convergence speed, while PG is more suitable for scenarios requiring absolute stability.

# 5. Preliminary investigations on Method Effectiveness

Based on systematic testing and analysis, we have drawn the following preliminary conclusions regarding the effectiveness of the three optimization methods:

First, all algorithms demonstrate good convergence on the Lasso problem, effectively recovering sparse signal patterns, which validates the mathematical correctness of the algorithm implementations. Specifically, PG exhibits the expected stable and monotonic convergence characteristics, with a convergence rate consistent with the theoretical $O(1/k)$ bound. FISTA indeed achieves accelerated convergence, with a significantly faster decline in the objective function during the early iterations compared to PG, approaching the theoretical $O(1/k^2)$ convergence rate, albeit with minor oscillations. S-FISTA's restart mechanism effectively prevents divergence, showing better numerical stability in complex problems.

In terms of computational efficiency, the three methods have similar per-iteration times, but significant differences in convergence speed, reflecting the classic trade-off between *computational efficiency* and *numerical stability*. These results fully demonstrate the effectiveness of our algorithm implementations and provide a reliable benchmark for subsequent research.

# 6. Project Improvement Recommendations

Based on the preliminary test findings, we propose the following improvements for the final project:

- **Expand Problem Diversity**: In addition to Lasso, include typical problems such as logistic regression and matrix completion to comprehensively evaluate the generalization capability of the algorithms.

- **Enhance Algorithm Robustness**: Implement an adaptive restart strategy in FISTA to dynamically adjust the momentum term based on gradient conditions, balancing convergence speed and stability.

- **Optimize Code Architecture**: Abstract common computational modules (line search, convergence judgment) into independent utility functions to reduce code redundancy and improve maintainability.
  
- **Improve Documentation**: Add detailed docstrings and usage examples, and establish an automated testing pipeline to ensure code quality and long-term maintainability.

- **Enable Large-Scale Computing**: Implement distributed computing support to handle very large-scale optimization problems, enhancing the practical value of the project.

These improvements will significantly enhance the robustness, usability, and scalability of our convex optimization toolbox.


# References

Beck, A., & Guttmann-Beck, N. (2018). FOM – a MATLAB toolbox of first-order methods for solving convex optimization problems. Optimization Methods and Software, 34(1), 172–193. https://doi.org/10.1080/10556788.2018.1437159  

Dossal, C., Hurault, S., & Papadakis, N. (2024). Optimization with first order algorithms. arXiv. https://doi.org/10.48550/arXiv.2410.19506
