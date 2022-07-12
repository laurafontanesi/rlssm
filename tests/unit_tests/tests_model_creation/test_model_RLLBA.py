import unittest

from rlssm.model.models_LBA import RLLBAModel_2A
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationRLLBA(unittest.TestCase):
    def test_model_creation_RLLBA(self, hier_levels=1):
        model_name = "RLLBA"

        rllba_model = RLLBAModel_2A(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)
