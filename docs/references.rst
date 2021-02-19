References
==========

The likelihood of the diffusion decision model, or DDM, (the Wiener first passage time distribution) is already implemented in stan (see stan's `manual`_).

The likelihood functions of the race models are implemented by us in stan, and are based on the following papers:

* Race diffusion model (RDM): Tillman, G., Van Zandtc, T., & Loganb, G. D. Psychon Bull Rev (2020), `RDM paper`_

* Linear ballistic accumulator (LBA): Brown, S. D., & Heathcote, A. Cognitive psychology (2008), `LBA paper`_

* Advantage LBA: van Ravenzwaaij, D., Brown, S. D., Marley, A. A. J., & Heathcote, A. Psychological review (2020), `ALBA paper`_

* Advantage RDM: Miletić, S., Boag, R. J., Trutti, A. C., Stevenson, N., Forstmann, B. U., & Heathcote, A. (2021), `ARDM paper`_

The RLDDMs (combinations of RL and DDM, diffusion decision model) are based on the following papers:

* Fontanesi, L., Gluth, S., Spektor, M.S. & Rieskamp, J. Psychon Bull Rev (2019), `RLDDM paper 1`_

* Fontanesi, L., Palminteri, S. & Lebreton, M. Cogn Affect Behav Neurosci (2019), `RLDDM paper 2`_

For a more in-depth explanation of the model's cognitive mechanisms and theories, please refer to **INSERT PREPRINT HERE**.

.. _manual: https://mc-stan.org/docs/2_19/functions-reference/wiener-first-passage-time-distribution.html
.. _RDM paper: https://doi.org/10.3758/s13423-020-01719-6
.. _LBA paper: https://doi.org/10.1016/j.cogpsych.2007.12.002
.. _ALBA paper: https://doi.org/10.1037/rev0000166
.. _ARDM paper: https://doi.org/10.7554/eLife.63055
.. _RLDDM paper 1: https://doi.org/10.3758/s13423-018-1554-2
.. _RLDDM paper 2: https://doi.org/10.3758/s13415-019-00723-1