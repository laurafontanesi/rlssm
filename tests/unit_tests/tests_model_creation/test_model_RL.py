import unittest

from rlssm.model.models_RL import RLModel_2A
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationRL(unittest.TestCase):
    def test_model_creation_RL(self, hier_levels=1):
        model_name = "RL"

        rl_model = RLModel_2A(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)
