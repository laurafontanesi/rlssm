.. currentmodule:: rlssm

ModelResults class for DDMs (or RLDDMs)
=======================================

There is one class to inspect model fits of **DDM or combinations of RL and DDM (fitted on choices and response times)**: :ref:`DDModelResults <DDModelResults>`.

The main functions of this class are:

* To assess the model's **convergence** and **mcmc diagnostics**, to make sure that the sampling was successful. This step is crucial and should be preferably done first.

* To provide a measure of the model's **quantitative fit** to the data (i.e., the Watanabe-Akaike information criterion). This is important when comparing the quantitative fit to the data of several, competing models.

* To visualize and make interval-based (either Bayesian Credible Intervals or Higher Density Intervals) inferences on the **posterior distributions** of the model's parameters. This is important when specific hypotheses were made about the parameters' values.

* To calculate and visualize **posterior predictive distributions** of the observed data. This step is important to assess the qualitative fit of the model to the data. Qualitative fit should be assessed not only when comparing different competing models, but also when a single candidate model is fitted. Different ways of calculating posterior predictive distributions are provided, together with different plotting options. In general, emphasis is given to calculating posterior predictive distributions across conditions. This allows us to assess whether a certain behavioral pattern observed in the data (e.g., due to experimental manipulations) can also be reproduced by the model. Finally, posterior predictives are available not only for mean choices and response times, but also for other summary statistics of the response times distributions (i.e., skewness and quantiles).

All models
----------

.. autoclass:: rlssm.fit.fits.ModelResults
    :members:

Diffusion decision models
-------------------------

.. _DDModelResults:
.. autoclass:: rlssm.fit.fits_DDM.DDModelResults
    :members:

    :show-inheritance:
    :inherited-members:

Reinforcement learning diffusion decision models
------------------------------------------------

See :ref:`DDModelResults <DDModelResults>`.