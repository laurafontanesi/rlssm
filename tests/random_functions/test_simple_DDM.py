import os

import numpy as np
import pandas as pd

from rlssm.random.random_DDM import simulate_ddm


# Sequential sampling models data
# NON HIER DATA
def test_simple_DDM(print_results=True):
    print("Test - simple simulate_ddm")

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
    assert np.mean(data1['accuracy']) > np.mean(data2['accuracy']), f"Accuracy(data1) is not better than accuracy(data2)"

    # TEST: check if data3 has slower response time than data1 and data3 has higher accuracy than data1
    assert np.mean(data3['rt']) > np.mean(data1['rt']), f"RT data3 is not slower than RT data1"
    assert np.mean(data3['accuracy']) > np.mean(data1['accuracy']), f"Accuracy(data3) is not better than accuracy(data1)"

    # TEST: threshold should be positive
    assert all(i > 0 for i in data1['threshold']), f"Threshold should be positive"

    # TEST: rel_sp should be positive
    assert all(i > 0 for i in data1['rel_sp']), f"rel_sp should be positive"

    # TEST: data4 response time for accurate decision is faster than for non accurate decision
    assert np.mean(data4.loc[data4['accuracy'] == 1.0]['rt']) <= np.mean(data4.loc[data4['accuracy'] == 0]['rt']), \
        f"rt(accurate decision) should be faster than rt(non accurate decision)"

    # TEST: rt for data 5 > rt for data 1
    assert np.mean(data5['rt']) > np.mean(data1['rt']), f"min rt(data5) should be greater than min rt(data1)"

    # Test data produced against reference data
    reference_path = os.path.join(os.path.dirname(__file__), 'reference_data', 'simple_DDM.csv')
    # data1.to_csv(reference_path)
    reference_data = pd.read_csv(reference_path, index_col=0)
    # assert data.equals(reference_data)
    