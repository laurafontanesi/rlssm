import unittest

from rlssm.model.models_ARDM import ARDModel_2A
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationARDM(unittest.TestCase):
    def test_model_creation_ARDM(self, hier_levels=1):
        model_name = "ARDM"

        ardm_model = ARDModel_2A(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)
