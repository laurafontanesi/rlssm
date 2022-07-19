import unittest

from rlssm.model.models_ARDM import RLARDModel_2A
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationRLARDM(unittest.TestCase):
    def test_model_creation_RLARDM(self, hier_levels=1):
        model_name = "RLARDM"

        rlardm_model = RLARDModel_2A(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)

        assert rlardm_model.priors is not None, "Priors of the model cannot be retrieved"
