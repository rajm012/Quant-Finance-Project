
from pymoo.core.problem import Problem

class BaseCMOP(Problem):
    """
    Base class for Constrained Multi-Objective Problems.
    All problems in this paper inherit from this class.
    """
    def __init__(self, n_var=None, n_obj=3, xl=0, xu=1, **kwargs):
        # Default n_var = n_obj + 4 (standard for DTLZ problems)
        if n_var is None:
            n_var = n_obj + 9
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        This method must be implemented by subclasses.
        It should compute the objective values and constraint violations.
        """
        raise NotImplementedError

    def get_pareto_front(self, n_points=10000):
        """
        Return the true Pareto front points (for IGD calculation).
        This method should be implemented by each problem.
        """
        raise NotImplementedError
    
    