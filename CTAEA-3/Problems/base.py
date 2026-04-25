
"""
Base class for all constrained multiobjective test problems.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseProblem(ABC):
    """
    Abstract base class for constrained multiobjective optimization problems.
    Attributes
    ----------
    n_var : int
        Number of decision variables.
    n_obj : int
        Number of objectives.
    n_con : int
        Number of constraints.
    xl : np.ndarray
        Lower bounds for decision variables.
    xu : np.ndarray
        Upper bounds for decision variables.
    """

    def __init__(self, n_var, n_obj, n_con, xl, xu):
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_con = n_con
        self.xl = np.array(xl)
        self.xu = np.array(xu)

    @abstractmethod
    def evaluate(self, X):
        """
        Evaluate objectives and constraints for a population X.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_var)
            Population of candidate solutions.

        Returns
        -------
        F : np.ndarray, shape (N, n_obj)
            Objective values.
        G : np.ndarray, shape (N, n_con)
            Constraint violation values (G[i,j] <= 0 means feasible).
        """
        pass

    def constraint_violation(self, G):
        """
        Compute total constraint violation (CV) per individual.
        CV = sum of positive constraint violations.
        The convention: G[i,j] > 0 means violated.

        Parameters
        ----------
        G : np.ndarray, shape (N, n_con)

        Returns
        -------
        CV : np.ndarray, shape (N,)
        """
        return np.sum(np.maximum(G, 0.0), axis=1)

    def sample_random(self, n):
        """
        Uniformly sample n random solutions in [xl, xu].
        """
        return np.random.uniform(self.xl, self.xu, size=(n, self.n_var))

