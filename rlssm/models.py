from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
from .stan_utility import compile_model

__dir__ = os.path.abspath(os.path.dirname(__file__))

class Model(object):
    """General model class.

    """
    def __init__(self, hierarchical_levels, family):
        """Initialize a Model object.

        Parameters
        ----------
        hierarchical_levels : int
            Set to 1 for individual data and to 2 for grouped data.

        family : str
            Model family. At the moment either "RL", "DDM", "RDM", or "RLDDM" "RLRDM".

        Attributes
        ----------
        model_label : str
            The label of the fully specified model.

        """
        if not isinstance(hierarchical_levels, int):
            raise TypeError("hierarchical_levels must be integer")
        if np.sum(hierarchical_levels == np.array([1, 2])) < 1:
            raise ValueError("set to 1 for individual data, and to 2 for grouped data")

        self.hierarchical_levels = hierarchical_levels
        self.family = family
        self.compiled_model = None

        if self.hierarchical_levels == 2:
            self.model_label = 'hier' + family
        else:
            self.model_label = family

    def _set_model_path(self):
        """Sets the stan model path.

        """
        stan_model_path = os.path.join(__dir__, 'stan_models', '{}.stan'.format(self.model_label))
        #stan_model_path = os.path.join(rlddm.__path__[0], 'stan_models', '{}.stan'.format(self.model_label))
        if not os.path.exists(stan_model_path):
            raise ValueError("Model {}, in {}, has not been implemented yet.".format(self.model_label, __dir__))
        self.stan_model_path = stan_model_path

    def _compile_stan_model(self):
        """Compiles the stan model.

        """
        self.compiled_model = compile_model(filename=self.stan_model_path,
                                            model_name=self.model_label)