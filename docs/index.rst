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
   cite
   authors

.. toctree::
   :maxdepth: 2
   :caption: How to:

   models/define_model
   models/inspect_simpleRL
   models/inspect_SSM_DDM
   models/inspect_SSM_race
   simulations/simulateDDM

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`