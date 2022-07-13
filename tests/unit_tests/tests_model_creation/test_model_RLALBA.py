import unittest

from rlssm.model.models_ALBA import RLALBAModel_2A
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationRLALBA(unittest.TestCase):
    def test_model_creation_RLALBA(self, hier_levels=1):
        model_name = "RLALBA"

        rlalba_model = RLALBAModel_2A(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)

        # Test if the priors can be retrieved
        _ = rlalba_model.priors
