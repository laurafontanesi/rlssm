import os

from tests.model_creation.test_ALBA_model import test_ALBA_model, test_hierALBA_model
from tests.model_creation.test_ARDM_model import test_ARDM_model, test_hierARDM_model
from tests.model_creation.test_DDM_model import test_DDM_model, test_hierDDM_model
from tests.model_creation.test_LBA_model import test_LBA_model, test_hierLBA_model
from tests.model_creation.test_RDM_model import test_RDM_model, test_hierRDM_model
from tests.model_creation.test_RLALBA_model import test_RLALBA_model, test_hierRLALBA_model
from tests.model_creation.test_RLARDM_model import test_RLARDM_model, test_hierRLARDM_model
from tests.model_creation.test_RLDDM_model import test_RLDDM_model, test_hierRLDDM_model
from tests.model_creation.test_RLLBA_model import test_RLLBA_model, test_hierRLLBA_model
from tests.model_creation.test_RLRDM_model import test_RLRDM_model, test_hierRLRDM_model
from tests.model_creation.test_RL_model import test_RL_model, test_hierRL_model


def test_model_creation(print_results=True):
    print("Running the model creation tests")
    print("----------------------------------")

    testing_dir_path = os.path.dirname(os.path.realpath(__file__))
    success_tests_ran = total_tests = len([item for item in os.listdir(testing_dir_path) if '.py' in item]) - 1

    pkl_path = os.path.join(os.path.dirname(os.getcwd()), 'pkl_files')
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    tests_to_run = [
        test_ALBA_model, test_hierALBA_model,
        test_ARDM_model, test_hierARDM_model,
        test_DDM_model, test_hierDDM_model,
        test_LBA_model, test_hierLBA_model,
        test_RDM_model, test_hierRDM_model,
        test_RL_model, test_hierRL_model,
        test_RLALBA_model, test_hierRLALBA_model,
        test_RLARDM_model, test_hierRLARDM_model,
        test_RLDDM_model, test_hierRLDDM_model,
        test_RLLBA_model, test_hierRLLBA_model,
        test_RLRDM_model, test_hierRLRDM_model
    ]

    for t in tests_to_run:
        try:
            t(print_results)
        except Exception as exc:
            print(f"Model creation tests: Exception occurred: {exc}")
            success_tests_ran -= 1

    print(f"Model creation tests: Succesfully ran {success_tests_ran}/{total_tests} tests")
    print("----------------------------------")
