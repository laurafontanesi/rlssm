.. rlssm documentation master file, created by
   sphinx-quickstart on Thu Feb 18 10:46:18 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rlssm's documentation
================================

rlssm is a Python package for fitting **reinforcement learning** (RL) models, **sequential sampling models** (DDM, RDM, LBA, ALBA, and ARDM), and **combinations of the two**, using **Bayesian parameter estimation**. 

Parameter estimation is done at an individual or hierarchical level using `PyStan`_, the Python Interface to Stan. Stan performs Bayesian inference using the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.

.. _PyStan: https://pypi.org/project/pystan/

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

   notebooks/load_from_saved

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   notebooks/DDM_fitting
   notebooks/DDM_hierarchical_fitting
   notebooks/DDM_starting-point-bias_fitting
   notebooks/DDM_starting-point-bias_hierarchical_fitting

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

Index
=====

* :ref:`genindex`