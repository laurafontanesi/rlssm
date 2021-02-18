Installing rlssm 
================

For now, you can simply install the rlssm package using ``python setup.py install``, after downloading or cloning rlssm from `GitHub`_.

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

	conda create --name rlssmenv python=3 pandas scipy seaborn pystan=2.19
	conda activate rlssmenv
	python setup.py install

Check if the compiler is working::

	python test.py

On MacOS, if you encounter a compilation error, you can try the following::

	conda create -n rlssmenv python=3.7 pandas cython seaborn scipy
	conda activate rlssmenv
	conda install clang_osx-64 clangxx_osx-64 -c anaconda
	conda info

Copy the "active env location"::

	ls ACTIVE_ENV_LOCATION/bin/ | grep clang | grep -i 'apple'

Copy the two clang and modify the following::

	export CC=x86_64-apple-darwin13.4.0-clang
	export CXX=x86_64-apple-darwin13.4.0-clang++

Now you can install pystan::

	conda install -c conda-forge pystan=2.19

And finally install the rlssm package and test again whether the compiler works::

	python setup.py install
	python test.py

If you want to try out also the notebooks::

	conda install -c conda-forge jupyterlab

If you are experiencing other issues with the compiler, check the `pystan documentation`_.

.. _pystan documentation: https://pystan.readthedocs.io/en/latest/