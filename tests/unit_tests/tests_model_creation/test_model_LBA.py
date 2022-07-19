import unittest

from rlssm.model.models_LBA import LBAModel_2A
from tests.unit_tests.tests_model_creation.helper_methods import check_pkl_file_existence


class TestModelCreationLBA(unittest.TestCase):
    def test_model_creation_LBA(self, hier_levels=1):
        model_name = "LBA"

        lba_model = LBAModel_2A(hierarchical_levels=hier_levels)

        check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)

        assert lba_model.priors is not None, "Priors of the model cannot be retrieved"
