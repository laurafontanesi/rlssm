import os

import matplotlib.pyplot as plt
from rlssm import RLModel_2A, load_example_dataset
from tests.plot_functions.common_methods import compute_MSE_error


def test_RL_plot_posterior(print_results=True):
    model = RLModel_2A(hierarchical_levels=1)

    data = load_example_dataset(hierarchical_levels=1)

    model_fit = model.fit(
        data,
        K=4,
        initial_value_learning=27.5,
        sensitivity_priors={'mu': 0, 'sd': 5},
        iter=2000,
        chains=2,
        verbose=False)

    model_fit.plot_posteriors(show_intervals="BCI")

    ref_fldr = os.path.join(os.path.dirname(__file__), 'reference_data')
    curr_image_path = os.path.join(ref_fldr, 'curr_img.png')
    ref_image_path = os.path.join(ref_fldr, 'ref_plot_post_rl.png')

    plt.savefig(curr_image_path)

    # TEST that the current plot was saved
    assert os.path.exists(curr_image_path), f"Failed checking existence of the plot"

    # TEST: check that difference to reference image is minimal
    reference_image = plt.imread(ref_image_path)
    current_image = plt.imread(curr_image_path)
    mse = compute_MSE_error(reference_image, current_image)
    assert mse < 0.5, f"The generated plot is too dissimilar with the reference plot"

    os.remove(curr_image_path)
