from rlssm import DDModel
from tests.model_creation.common_methods import check_pkl_file_existence


def test_DDM_model(hier_levels=1, print_results=True):
    model_name = "DDM"

    ddm_model = DDModel(hierarchical_levels=hier_levels)

    check_pkl_file_existence(model_name=model_name, hier_levels=hier_levels)
