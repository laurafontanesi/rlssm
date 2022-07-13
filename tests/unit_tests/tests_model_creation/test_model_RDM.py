import unittest

from rlssm.model.models_RDM import RDModel_2A
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationRDM(unittest.TestCase):
    def test_model_creation_RDM(self, hier_levels=1):
        model_name = "RDM"

        rdm_model = RDModel_2A(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)

        # Test if the priors can be retrieved
        _ = rdm_model.priors
