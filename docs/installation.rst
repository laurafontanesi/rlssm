Installation 
============

You can install the rlssm package using ``pip install rlssm``, or get it directly from `GitHub`_.

Make sure you have the dependencies installed first.

.. _Github: https://github.com/laurafontanesi/rlssm

Dependencies
------------
- cmdstanpy=1.0.4
- pandas
- scipy
- seaborn

Quick start
-----------------------------

In order to install **cmdstanpy**, check the `cmdstanpy documentation`_.

.. _cmdstanpy documentation: https://cmdstanpy.readthedocs.io/en/stable-0.9.65/getting_started.html

This block of code checks if the package and its dependencies are installed, displaying a success or fail message.

::

    import rlssm

    model = rlssm.DDModel(hierarchical_levels=2)
    priors = model.priors
    print(priors)

    if (i in str(priors) for i in ["drift_priors", "ndt_priors", "mu_mu", "sd_sd"]):
        print("Compiler test results: SUCCESS")
    else:
        print("Compiler test results: FAIL")

As an alternative, run the compiler test by calling the function doing the following:

::

        from rlssm.tests import test_compiler
        test_compiler()