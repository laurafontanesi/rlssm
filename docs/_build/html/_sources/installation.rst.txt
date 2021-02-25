Installation 
============

You can install the rlssm package using ``pip install rlssm``, or get it directly from `GitHub`_.

Make sure you have the dependecies installed first.

.. _Github: https://github.com/laurafontanesi/rlssm

Dependencies
------------
- pystan=2.19
- pandas
- scipy
- seaborn

Conda environment (suggested)
-----------------------------

If you have Andaconda or miniconda installed and you would like to create a separate environment for the rlssm package, do the following::

	conda create --n stanenv python=3 pandas scipy seaborn pystan=2.19
	conda activate stanenv
	python setup.py install

Check if the compiler is working::

	python tests/test_compiler.py

On MacOS, if you encounter a compilation error, you can try the following::

	conda create -n stanenv python=3.7 pandas cython seaborn scipy
	conda activate stanenv
	conda install clang_osx-64 clangxx_osx-64 -c anaconda
	conda info

Copy the "active env location" and substitute into::

	ls ACTIVE_ENV_LOCATION/bin/ | grep clang | grep -i 'apple'

Copy the two clangs and modify the following::

	export CC=x86_64-apple-darwin13.4.0-clang
	export CXX=x86_64-apple-darwin13.4.0-clang++

Install pystan and rlssm and test again whether the compiler works::

	conda install -c conda-forge pystan=2.19
	python setup.py install
	python tests/test_compiler.py

If you are experiencing other issues with the compiler, check the `pystan documentation`_.

.. _pystan documentation: https://pystan.readthedocs.io/en/latest/