import numpy as np

from rlssm.random.random_DDM import simulate_ddm, simulate_hier_ddm
from rlssm.random.random_common import generate_task_design_fontanesi


# Sequential sampling models data
def test_random_DDM(print_results=True):
    data1 = simulate_ddm(
        n_trials=1000,
        gen_drift=.6,
        gen_drift_trialsd=.1,
        gen_threshold=1.4,
        gen_ndt=.23)

    data2 = simulate_ddm(
        n_trials=1000,
        gen_drift=-.6,
        gen_drift_trialsd=.1,
        gen_threshold=1.4,
        gen_ndt=.23)

    data3 = simulate_ddm(
        n_trials=1000,
        gen_drift=.6,
        gen_drift_trialsd=.1,
        gen_threshold=2.4,
        gen_ndt=.23)

    data4 = simulate_ddm(
        n_trials=1000,
        gen_drift=.6,
        gen_drift_trialsd=.1,
        gen_rel_sp=.8,
        gen_threshold=1.4,
        gen_ndt=.23)

    data5 = simulate_ddm(
        n_trials=1000,
        gen_drift=.6,
        gen_drift_trialsd=.1,
        gen_threshold=1.4,
        gen_ndt=.5)

    # TEST: assure there is only 1 participant
    assert data1.index[-1][0] == 1, f"Number of participants should be 1"

    # TEST: check that accuracy of data1 > accuracy of data2
    assert np.mean(data1['accuracy']) > np.mean(
        data2['accuracy']), f"Accuracy(data1) is not better than accuracy(data2)"

    # TEST: check if data3 has slower response time than data1 and data3 has higher accuracy than data1
    assert np.mean(data3['rt']) > np.mean(data1['rt']), f"RT data3 is not slower than RT data1"
    assert np.mean(data3['accuracy']) > np.mean(
        data1['accuracy']), f"Accuracy(data3) is not better than accuracy(data1)"

    # TEST: threshold should be positive
    assert all(i > 0 for i in data1['threshold']), f"Threshold should be positive"

    # TEST: rel_sp should be positive
    assert all(i > 0 for i in data1['rel_sp']), f"rel_sp should be positive"

    # TEST: data4 response time for accurate decision is faster than for non accurate decision
    assert np.mean(data4.loc[data4['accuracy'] == 1.0]['rt']) <= np.mean(data4.loc[data4['accuracy'] == 0]['rt']), \
        f"rt(accurate decision) should be faster than rt(non accurate decision)"

    # TEST: rt for data 5 > rt for data 1
    assert np.mean(data5['rt']) > np.mean(data1['rt']), f"min rt(data5) should be greater than min rt(data1)"

    # TEST hierarchical version
    dm = generate_task_design_fontanesi(n_trials_block=80,
                                        n_blocks=3,
                                        n_participants=30,
                                        trial_types=['1-2', '1-3', '2-4', '3-4'],
                                        mean_options=[34, 38, 50, 54],
                                        sd_options=[5, 5, 5, 5])

    data_hier = simulate_hier_ddm(n_trials=100,
                                  n_participants=30,
                                  gen_mu_drift=1,
                                  gen_sd_drift=.5,
                                  gen_mu_threshold=1,
                                  gen_sd_threshold=.1,
                                  gen_mu_ndt=.23,
                                  gen_sd_ndt=.1)

    # TEST: assure that there are 30 participants
    assert data_hier.index[-1][0] == 30, f"Number of participants should be 30"
