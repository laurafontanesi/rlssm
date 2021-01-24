from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

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
description = "rlssm: a package for fitting RL models, DDM, and combinations of the two"
# Long description will go up on the pypi page
long_description = """

rlssm
========
rlssm is a Python package for fitting RL models, DDM, and combinations of the two using Bayesian parameter estimation and it's based on [PyStan](https://pystan.readthedocs.io/en/latest/index.html).

The RLSSMs (combinations of RL and DDM) are based on the following papers:
- Fontanesi, L., Gluth, S., Spektor, M.S. et al. Psychon Bull Rev (2019) 26: 1099. https://doi.org/10.3758/s13423-018-1554-2
- Fontanesi, L., Palminteri, S. & Lebreton, M. Cogn Affect Behav Neurosci (2019) 19: 490. https://doi.org/10.3758/s13415-019-00723-1

License
=======
``rlssm`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2019--, Laura Fontanesi,
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
