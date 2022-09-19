.. rlssm documentation master file, created by
   sphinx-quickstart on Thu Feb 18 10:46:18 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rlssm's documentation
================================

`rlssm` is a Python package for fitting **reinforcement learning** (RL) models, **sequential sampling models** (DDM, RDM, LBA, ALBA, and ARDM), and **combinations of the two**, using **Bayesian parameter estimation**. 

Parameter estimation is done at an individual or hierarchical level using `CmdPyStan`_, the Python interface to CmdStan, which provides access to the Stan compiler. Stan performs Bayesian inference using the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.

.. _CmdPyStan: https://pypi.org/project/cmdstanpy/1.0.4/


.. image:: _static/identicons.png

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   installation
   references
   credits

.. toctree::
   :maxdepth: 2
   :caption: How to:

   notebooks/initialize_model
   notebooks/fit_model
   notebooks/inspect_model

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   notebooks/DDM_fitting
   notebooks/DDM_hierarchical_fitting
   notebooks/DDM_starting-point-bias_fitting
   notebooks/DDM_starting-point-bias_hierarchical_fitting
   notebooks/RL_2A_fitting
   notebooks/RL_2A_hierarchical_fitting
   notebooks/RLDDM_fitting
   notebooks/LBA_2A_fitting

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   models/model_classes
   models/modelresult_RL
   models/modelresult_DDM
   models/modelresult_race
   simulations/simulateDDM
   simulations/simulaterace
   simulations/simulateRL

.. toctree::
   :maxdepth: 1
   :caption: Updates over time:

   changelist

Index
=====

* :ref:`genindex`