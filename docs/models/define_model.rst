Define and fit a model
======================

.. currentmodule:: rlssm

These classes can be used to define different available models. There are three main classes, corresponding to the three main classes of models that are implemented in `rlddm`: :ref:`RLModel <RLModel>`, :ref:`DDModel <DDModel>`, and :ref:`RLDDModel <RLDDModel>`.

All classes have a hierarchical and non-hierarchical version, and come with additional cognitive mechanisms that can be added or excluded.

Reinforcement learning models
-----------------------------

.. _RLModel:
.. autoclass:: RLModel
    :members:

    .. automethod:: __init__

Diffusion decision models
-------------------------

.. _DDModel:
.. autoclass:: DDModel
    :members:

    .. automethod:: __init__

Reinforcement learning diffusion decision models
------------------------------------------------

.. _RLDDModel:
.. autoclass:: RLDDModel
    :members:

    .. automethod:: __init__
