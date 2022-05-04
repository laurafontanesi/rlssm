import unittest

from rlssm.model.models_DDM import RLDDModel
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationRLDDM(unittest.TestCase):
    def test_model_creation_RLDDM(self, hier_levels=1):
        model_name = "RLDDM"

        rlddm_model = RLDDModel(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)
