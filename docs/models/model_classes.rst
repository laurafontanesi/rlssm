Model classes
=============

.. currentmodule:: rlssm

These classes can be used to define different available models. Currently, 5 classes of models are implemented in `rlssm`:

1. simple reinforcement learning models: :ref:`RLModel_2A <RLModel_2A>`
2. diffusion decision models: :ref:`DDModel <DDModel>`
3. reinforcement learning diffusion decision models: :ref:`RLDDModel <RLDDModel>`
4. race models: :ref:`RDModel_2A <RDModel_2A>`, :ref:`LBAModel_2A <LBAModel_2A>`, :ref:`ARDModel_2A <ARDModel_2A>`, :ref:`ALBAModel_2A <ALBAModel_2A>`
5. reinforcement learning race models: :ref:`RLRDModel_2A <RLRDModel_2A>`, :ref:`RLLBAModel_2A <RLLBAModel_2A>`, :ref:`RLARDModel_2A <RLARDModel_2A>`, :ref:`RLALBAModel_2A <RLALBAModel_2A>`

All classes have a hierarchical and non-hierarchical version, and come with additional cognitive mechanisms that can be added or excluded.

.. note:: At the moment, all model classes are meant for decisions between 2 alternatives.

Reinforcement learning models (for 2 alternatives)
--------------------------------------------------

.. _RLModel_2A:
.. autoclass:: RLModel_2A
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

Race models (for 2 alternatives)
--------------------------------

.. _RDModel_2A:
.. autoclass:: RDModel_2A
    :members:

    .. automethod:: __init__

.. _LBAModel_2A:
.. autoclass:: LBAModel_2A
    :members:

    .. automethod:: __init__

.. _ARDModel_2A:
.. autoclass:: ARDModel_2A
    :members:

    .. automethod:: __init__

.. _ALBAModel_2A:
.. autoclass:: ALBAModel_2A
    :members:

    .. automethod:: __init__

Reinforcement learning race models (for 2 alternatives)
-------------------------------------------------------

.. _RLRDModel_2A:
.. autoclass:: RLRDModel_2A
    :members:

    .. automethod:: __init__

.. _RLLBAModel_2A:
.. autoclass:: RLLBAModel_2A
    :members:

    .. automethod:: __init__

.. _RLARDModel_2A:
.. autoclass:: RLARDModel_2A
    :members:

    .. automethod:: __init__

.. _RLALBAModel_2A:
.. autoclass:: RLALBAModel_2A
    :members:

    .. automethod:: __init__