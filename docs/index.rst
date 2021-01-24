.. rlssm documentation master file, created by
   sphinx-quickstart on Thu Feb 20 19:08:17 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rlssm's documentation
================================

`rlssm` is a Python package for fitting reinforcement learning (RL) models, sequential sampling models (SSM), and combinations of the two (RLSSM).

Parameter estimation is done at an individual or hierarchical level using `PyStan`_, the Python Interface to Stan. Stan performs Bayesian inference using the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.

.. _PyStan: https://pystan.readthedocs.io/

.. image:: _static/identicons.png

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   installation

.. toctree::
   :maxdepth: 2
   :caption: How to:

   models/define_model
   models/inspect_model
   simulations/simulateDDM
