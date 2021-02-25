from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = 1  # use '' for first of series, number for 1 and above
_version_extra = 'dev'  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "rlssm: a package for fitting RL models, SSM, and combinations of the two"
# Long description will go up on the pypi page
long_description = """

rlssm
========
``rlssm`` is a Python package for fitting reinforcement learning (RL) models,
sequential sampling models (DDM, RDM, LBA, ALBA, and ARDM),
and combinations of the two, using Bayesian parameter estimation.

Parameter estimation is done at an individual or hierarchical level
using ``PyStan``, the Python Interface to Stan.
Stan performs Bayesian inference using the No-U-Turn sampler,
a variant of Hamiltonian Monte Carlo.

Documentation
=============
The latest documentation can be found here: https://rlssm.readthedocs.io/

License
=======
``rlssm`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2021--, Laura Fontanesi,
University of Basel.
"""

NAME = "rlssm"
MAINTAINER = "Laura Fontanesi"
MAINTAINER_EMAIL = "laura.fontanesi.1@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/laurafontanesi/rlssm"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Laura Fontanesi"
AUTHOR_EMAIL = "laura.fontanesi.1@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'rlssm': [pjoin('data', '*')]}
REQUIRES = ["numpy", "pandas", "pystan"]