Changelist
============

This is a list of changes between different versions of the `rlssm` package.

v0.1.2 (01 October 2022)
------------------------
    - Use CmdStanPy instead of PyStan as the tool to interact with Stan, to achieve better performance and allow multi-platform usage of the package.
    - Reorganized the structure of the package to make it more user-friendly and easier to maintain.
    - Developed a robust test system, based on unittests, to test functionalities such as model creation, fitting, plotting, etc.

v0.1.1 (25 February 2021)
-------------------------
    - Added models: RL, DDM, LBA, ALBA, RDM, ARDM, RL+(DDM, LBA, ALBA, RDM, ARDM) and their hierarchical version.
    - Added functions for fitting, model creation, results plotting, and simulating for each model.
    - Added functionality to save and reload the models using pickle.
    - Use PyStan as the main tool to interact with Stan.
    - Used sphinx to build the documentation (https://rlssm.readthedocs.io/en/latest/).