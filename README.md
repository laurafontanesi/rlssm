# rlssm

rlssm is a Python package for fitting **reinforcement learning** (RL) models, **sequential sampling models** (DDM, RDM, LBA, ALBA, and ARDM), and **combinations of the two**, using **Bayesian parameter estimation**. 

Parameter estimation is done at an individual or hierarchical level using [PyStan](https://pystan.readthedocs.io/en/latest/index.html), the Python Interface to Stan. Stan performs Bayesian inference using the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo.

## Install
You can install the rlssm package using: 
```
pip install rlssm
```

Make sure you have the dependecies installed first.

### Dependencies
- pystan=2.19
- pandas
- scipy
- seaborn

### Conda environment (suggested)
If you have Andaconda or miniconda installed and you would like to create a separate environment:

```
conda create --n stanenv python=3 pandas scipy seaborn pystan=2.19
conda activate stanenv
pip install rlssm
```
## Documentation

The latest documentation can be found here: https://rlssm.readthedocs.io/

## Cite

[![DOI](https://zenodo.org/badge/332414951.svg)](https://zenodo.org/badge/latestdoi/332414951)
