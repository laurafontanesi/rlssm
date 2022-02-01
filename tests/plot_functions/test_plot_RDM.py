import os

import matplotlib.pyplot as plt
from rlssm import load_example_dataset, RDModel_2A
from tests.plot_functions.common_methods import compute_MSE_error


def test_RDM_plot_posterior(print_results=True):
    model = RDModel_2A(hierarchical_levels=1)

    data = load_example_dataset(hierarchical_levels=1)

    threshold_priors = {'mu': 0, 'sd': 5}
    drift_priors = {'mu': 0, 'sd': 5}
    ndt_priors = {'mu': 0, 'sd': 5}

    model_fit = model.fit(data,
                          threshold_priors=threshold_priors,
                          ndt_priors=drift_priors,
                          drift_priors=ndt_priors,
                          iter=1000,
                          chains=2,
                          pointwise_waic=False,
                          verbose=False,
                          print_diagnostics=print_results)

    model_fit.plot_posteriors(show_intervals="BCI")

    ref_fldr = os.path.join(os.path.dirname(__file__), 'reference_data')
    curr_image_path = os.path.join(ref_fldr, 'curr_img.png')
    ref_image_path = os.path.join(ref_fldr, 'ref_plot_post_rdm.png')

    plt.savefig(curr_image_path)

    # TEST that the current plot was saved
    assert os.path.exists(curr_image_path), f"Failed checking existence of the plot"

    # TEST: check that difference to reference image is minimal
    reference_image = plt.imread(ref_image_path)
    current_image = plt.imread(curr_image_path)
    mse = compute_MSE_error(reference_image, current_image)
    assert mse < 0.5, f"The generated plot is too dissimilar with the reference plot"

    os.remove(curr_image_path)
