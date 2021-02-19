Simulate data with the DDM
==========================

.. currentmodule:: rlssm.random

These functions can be used to simulate data of a single participant or of a group of participants, given a set of parameter values.

These functions can be thus used for parameter recovery: A model can be fit on the simulated data in order to compare the generating parameters with their estimated posterior distributions. For such purpose, :ref:`simulate_ddm <simulate_ddm>` (for a single participant) and :ref:`simulate_hier_ddm <simulate_hier_ddm>` (for a group of participants) should be used.

For faster calculations, parameters can be given as numpy.ndarrays to :ref:`random_ddm <random_ddm>` and :ref:`random_ddm_vector <random_ddm_vector>`.

In pandas
---------

.. _simulate_ddm:
.. autofunction:: simulate_ddm

.. _simulate_hier_ddm:
.. autofunction:: simulate_hier_ddm

In numpy
--------

.. _random_ddm:
.. autofunction:: random_ddm

.. _random_ddm_vector:
.. autofunction:: random_ddm_vector