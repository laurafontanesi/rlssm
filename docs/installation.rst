Installation 
============

You can install the rlssm package using ``pip install rlssm``, or get it directly from `GitHub`_.

Make sure you have the dependencies installed first.

.. _Github: https://github.com/laurafontanesi/rlssm

Dependencies
------------
- cmdstanpy>=1.0.4
- numpy
- pandas
- seaborn
- scipy
- matplotlib
- setuptools

Quick start
-----------------------------

In order to install **cmdstanpy**, check the `cmdstanpy documentation`_.

.. _cmdstanpy documentation: https://cmdstanpy.readthedocs.io/en/stable-0.9.65/getting_started.html

This block of code checks if the package and its dependencies are installed.

::

    from rlssm.model.models_DDM import DDModel

    model = DDModel(hierarchical_levels=2)
    priors = model.priors
    assert model.priors is not None, "Priors of the model cannot be retrieved"
    print(priors)

As an alternative, run the compiler test by calling the function doing the following:

::

        from rlssm import test_compiler
        test_compiler()

If installed correctly, the output should be:
::

    {'drift_priors': {'mu_mu': 1, 'sd_mu': 5, 'mu_sd': 0, 'sd_sd': 5}, 'threshold_priors': {'mu_mu': 1, 'sd_mu': 3, 'mu_sd': 0, 'sd_sd': 3}, 'ndt_priors': {'mu_mu': 1, 'sd_mu': 1, 'mu_sd': 0, 'sd_sd': 1}}
