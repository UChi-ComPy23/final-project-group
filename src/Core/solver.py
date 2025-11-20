"""
Base class for all optimization solvers.
All solvers inherit from this class.
This class provides:
- storage of the problem oracle (ProblemBase)
- storage of the current iterate x_k
- iteration counter k
- history recording
- generic run() method
- abstract step() and stop() methods to override

Each solver implements its own iteration rule in step().
"""

class SolverBase:
    """Base class for optimization solvers.
    """

    def __init__(self, problem, x0):
        """
        Parameters
        ----------
        problem : ProblemBase
            The optimization problem supplying oracle functions.
        x0 : array-like
            Initial iterate.
        """
        self.problem = problem
        self.x = x0
        self.k = 0
        self.history = {}

    def step(self):
        """
        Perform a single solver iteration.
        Implemented in subclass.
        """
        raise NotImplementedError

    def stop(self):
        """
        Stopping Condition.
        Check for convergence.
        Default returns False (run until max_iter).
        Subclasses may override with different stopping condition.
        """
        return False

    def record(self, **kwargs):
        """
        Record diagnostic quantities.
        e.g. self.record(obj=f, step_size=alpha, residual=r)
        Automatically stores lists under history for future usage.
        """
        for key, value in kwargs.items():
            self.history.setdefault(key, []).append(value)

    def run(self, max_iter=1000):
        """
        Run the solver for up to max_iter iterations or until stop() is True.
        Return the final iterate.
        """
        for _ in range(max_iter):
            self.step()
            if self.stop():
                break
            self.k += 1
        return self.x