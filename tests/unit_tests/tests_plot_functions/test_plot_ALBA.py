import os
import unittest

import numpy as np
from matplotlib import pyplot as plt

from rlssm.model.models_ALBA import ALBAModel_2A
from rlssm.utility.load_data import load_example_dataset
from tests.unit_tests.tests_plot_functions.helper_methods import compute_MSE_error


class TestPlotALBA(unittest.TestCase):
    def test_plot_posterior_ALBA(self):
        model = ALBAModel_2A(hierarchical_levels=1)

        data = load_example_dataset(hierarchical_levels=1)

        data['S_cor'] = np.random.normal(.4, 0.01, data.shape[0])
        data['S_inc'] = np.random.normal(.3, 0.01, data.shape[0])

        model_fit = model.fit(data,
                              thin=5,
                              iter_sampling=500,
                              iter_warmup=500,
                              chains=2,
                              parallel_chains=2)

        model_fit.plot_posteriors(show_intervals="BCI")

        ref_fldr = os.path.join(os.path.dirname(__file__), 'reference_data')
        curr_image_path = os.path.join(ref_fldr, 'curr_img.png')
        ref_image_path = os.path.join(ref_fldr, 'ref_plot_post_alba.png')

        plt.savefig(curr_image_path)

        # TEST that the current plot was saved
        assert os.path.exists(curr_image_path), f"Failed checking existence of the plot"

        # TEST: check that difference to reference image is minimal
        reference_image = plt.imread(ref_image_path)
        current_image = plt.imread(curr_image_path)
        mse = compute_MSE_error(reference_image, current_image)
        assert mse < 0.5, f"The generated plot is too dissimilar with the reference plot"

        os.remove(curr_image_path)
