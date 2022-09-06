.. currentmodule:: rlssm

Simulate data with RL models, RLDDMs, and RL+race models
========================================================

These functions can be used to simulate data of a single participant or of a group of participants, given a set of parameter values.

These functions can be thus used for parameter recovery: A model can be fit on the simulated data in order to compare the generating parameters with their estimated posterior distributions.

.. note:: At the moment, only non-hierarchical RLRDM data can be simulated.

Simulate RL stimuli
-------------------

.. _generate_task_design_fontanesi:
.. autofunction:: rlssm.random.random_common.generate_task_design_fontanesi

Simulate only-choices RL data
-----------------------------

.. _simulate_rl_2A:
.. autofunction:: rlssm.random.random_RL.simulate_rl_2A

.. _simulate_hier_rl_2A:
.. autofunction:: rlssm.random.random_RL.simulate_hier_rl_2A

Simulate RLDDM data (choices and RTs)
-------------------------------------

.. _simulate_rlddm_2A:
.. autofunction:: rlssm.random.random_RL_DDM.simulate_rlddm_2A

.. _simulate_simulate_hier_rlddm_2A:
.. autofunction:: rlssm.random.random_RL_DDM.simulate_hier_rlddm_2A

Simulate RLRDM data (choices and RTs)
-------------------------------------

.. _simulate_rlrdm_2A:
.. autofunction:: rlssm.random.random_RL_RDM.simulate_rlrdm_2A

.. _simulate_hier_rlrdm_2A:
.. autofunction:: rlssm.random.random_RL_RDM.simulate_hier_rlrdm_2A