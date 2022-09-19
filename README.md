# rlssm

rlssm is a Python package for fitting **reinforcement learning** (RL) models, 
**sequential sampling models** (DDM, RDM, LBA, ALBA, and ARDM), and **combinations of the two**, 
using **Bayesian parameter estimation**. 

Parameter estimation is done at an individual or hierarchical level using 
[CmdStanPy](https://cmdstanpy.readthedocs.io/en/stable-0.9.65/index.html), the Interface to Stan. 
Stan performs Bayesian inference using the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.

## Install
The rlssm package can be installed by using: 
```
pip install rlssm
```

Make sure you have the dependencies installed first.

### Dependencies
- cmdstanpy
- numpy
- pandas
- seaborn
- scipy
- matplotlib
- setuptools

### Conda environment (suggested)
If you have Anaconda or miniconda installed, and you would like to create a separate environment:

```
conda create --name stanenv python=3 numpy pandas seaborn scipy matplotlib setuptools
conda activate stanenv
pip install --upgrade cmdstanpy
pip install --upgrade rlssm
```
## Documentation

The latest documentation can be found here: https://rlssm.readthedocs.io/

## Cite

[![DOI](https://zenodo.org/badge/332414951.svg)](https://zenodo.org/badge/latestdoi/332414951)
