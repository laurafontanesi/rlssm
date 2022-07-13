import unittest

from rlssm.model.models_ALBA import ALBAModel_2A
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationALBA(unittest.TestCase):
    def test_model_creation_ALBA(self, hier_levels=1):
        model_name = "ALBA"

        alba_model = ALBAModel_2A(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)

        # Test if the priors can be retrieved
        _ = alba_model.priors
