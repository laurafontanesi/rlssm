from rlssm import RLRDModel_2A
from tests.model_creation.common_methods import check_pkl_file_existence


def test_RLRDM_model(hier_levels=1, print_results=True):
    model_name = "RLRDM"

    rlrdm_model = RLRDModel_2A(hierarchical_levels=hier_levels)

    check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)
