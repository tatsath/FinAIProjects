.. _supervised:

.. automodule:: stable_baselines.common.policies

Supervised Learning
===============

Table containing list of Projects with the following columns
- project
- description
- github repository
- Google colab `Title <http://link>`_
- binder link
- FinAILab rating


.. note::

	CnnPolicies are for images only. MlpPolicies are made for other type of features (e.g. robot joints)

.. warning::
  For all algorithms (except DDPG, TD3 and SAC), continuous actions are clipped during training and testing
  (to avoid out of bound error).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    MlpLstmPolicy
    MlpLnLstmPolicy
    CnnPolicy
    CnnLstmPolicy
    CnnLnLstmPolicy


Base Classes
------------

.. autoclass:: BasePolicy
  :members:

.. autoclass:: ActorCriticPolicy
  :members:

.. autoclass:: FeedForwardPolicy
  :members:

.. autoclass:: LstmPolicy
  :members:

MLP Policies
------------

.. autoclass:: MlpPolicy
  :members:

.. autoclass:: MlpLstmPolicy
  :members:

.. autoclass:: MlpLnLstmPolicy
  :members:


CNN Policies
------------

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: CnnLstmPolicy
  :members:

.. autoclass:: CnnLnLstmPolicy
  :members:
