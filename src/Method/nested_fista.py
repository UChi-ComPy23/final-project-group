from core.solver import SolverBase

"""
Nested FISTA
Minimization model: minimize φ(f(x)) + λ g(Ax)

Assumptions:
-φ is Lipschitz, non-decreasing, and proximable
-f is smooth
-g is proper, closed, proximable
-λ > 0

Oracles:
-φ(x), prox_{αφ}(x)
-f(x), ∇f(x)
-g(x), prox_{αg}(x)
-A(x), A^T(y)
"""


class NestedFISTA(SolverBase):
    def __init__(self, problem, x0, lam, inner_solver, M0=1.0):
        
        super().__init__(problem, x0)

        self.x = x0.copy()
        self.y = x0.copy()
        self.t = 1.0
        self.t_prev = 1.0
        self.x_prev = x0.copy()

        self.lam = lam
        self.M = float(M0)
        self.inner_solver = inner_solver


    def step(self):
        """Perform one Nested FISTA update (paper exact form)."""

        # internal solver handles prox of g & φ composition
        z_k = self.inner_solver(self.problem, self.y, self.lam, self.M)

        def F(x):
            Ax = self.problem.A(x)
            return self.problem.phi(self.problem.f(x)) + self.lam*self.problem.g(Ax)

        x_new = z_k if F(z_k) <= F(self.x) else self.x.copy()
        t_new = 0.5*(1 + np.sqrt(1 + 4*self.t*self.t))
        y_new = (x_new
                 + (self.t/t_new)*(z_k - x_new)
                 + (self.t_prev/t_new)*(x_new - self.x))
        self.x_prev = self.x
        self.x = x_new
        self.t_prev = self.t
        self.t = t_new
        self.y = y_new
        self.record(obj=F(self.x))

	
