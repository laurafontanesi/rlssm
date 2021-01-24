Installing rlssm
================

With git and conda
------------------

You can clone rlssm using git.

To be sure that you have all the required libraries installed, the safest way is to create a separate conda environment (in this case also called rlssm), containing python 3, pystan 2.19, pandas, scipy and seaborn, and activate it ::

    conda create --name rlssm python=3 pystan=2.19 pandas scipy seaborn
    conda activate rlssm

    python setup.py install
