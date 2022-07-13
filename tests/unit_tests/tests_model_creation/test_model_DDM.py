import unittest

from rlssm.model.models_DDM import DDModel
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationDDM(unittest.TestCase):
    def test_model_creation_DDM(self, hier_levels=1):
        model_name = "DDM"

        ddm_model = DDModel(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)

        # Test if the priors can be retrieved
        _ = ddm_model.priors
