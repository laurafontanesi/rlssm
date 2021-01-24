# rlssm

rlssm is a Python package for fitting reinforcement learning (RL) models, diffusion and race diffusion models, and combinations of the two using Bayesian parameter estimation and it's based on [PyStan](https://pystan.readthedocs.io/en/latest/index.html).

The RLDDMs (combinations of RL and DDM, diffusion decision model) are based on the following papers:
- Fontanesi, L., Gluth, S., Spektor, M.S. & Rieskamp, J. Psychon Bull Rev (2019) https://doi.org/10.3758/s13423-018-1554-2
- Fontanesi, L., Palminteri, S. & Lebreton, M. Cogn Affect Behav Neurosci (2019) https://doi.org/10.3758/s13415-019-00723-1

The RLRDMs (combinations of RL and RDM, race diffusion model) are based on the following paper:
- Tillman, G., Van Zandtc, T., & Loganb, G. D. Psychon Bull Rev (2020) https://doi.org/10.3758/s13423-020-01719-6

The RLLBAs (combinations of RL and LBA, linear ballistic accumulator model) are based on the following paper:
- Brown, S. D., & Heathcote, A. Cognitive psychology (2008) https://doi.org/10.1016/j.cogpsych.2007.12.002

The RLARDMs and RLALBAs (combinations of RL and ARDM or ALBA) are based on the following papers:
- van Ravenzwaaij, D., Brown, S. D., Marley, A. A. J., & Heathcote, A. Psychological review (2020) https://doi.org/10.1037/rev0000166
- Miletic, S., Boag, R. J., Trutti, A. C., Forstmann, B. U., & Heathcote, A. bioRxiv (2020) https://doi.org/10.1101/2020.09.12.294512

## Installation
For now, you can simply install the rlssm package using `python setup.py install`, after dowloading/cloning the developer version from GitHub.

Make sure you have the dependecies installed first.

**Dependencies:**
- pystan=2.19
- pandas
- scipy
- seaborn

**With conda**:
If you want to create a separate conda environment for the rlssm package, do the following:
`conda create --name rlssmenv python=3 pandas scipy seaborn pystan=2.19`
`conda activate rlssmenv`
`python setup.py install`
(check if the compiler is working)
`python test.py`

On MacOS, if you encounter a compilation error, you can try the following:
`conda create -n rlssmenv python=3.7 pandas cython seaborn scipy`
`conda activate rlssmenv`
`conda install clang_osx-64 clangxx_osx-64 -c anaconda`
`conda info`
(copy the "active env location")
`ls ACTIVE_ENV_LOCATION/bin/ | grep clang | grep -i 'apple'`
(copy the two clang and modify the following)
`export CC=x86_64-apple-darwin13.4.0-clang`
`export CXX=x86_64-apple-darwin13.4.0-clang++`
(now you can install pystan)
`conda install -c conda-forge pystan=2.19`
(and finally install the rlssm package and test the compiler)
`python setup.py install`
`python test.py`
(if you want to try out also the notebooks)
`conda install -c conda-forge jupyterlab`

If you are experiencing issues with the compiler, check the pystan documentation: https://pystan.readthedocs.io/en/latest/

## Documentation

The latest documentation can be found:
