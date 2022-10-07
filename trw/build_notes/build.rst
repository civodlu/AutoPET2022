Conda requirement file for readthedocs
--------------------------------------

we use conda cpu to limit the size of the download else we reach the 1gb limit on RTD.

Mypy
----

To run mypy:

::

    mypy src --allow-redefinition

Cyclic dependencies isolation
-----------------------------

Use sphinx to locate the cycles. Run in an environment with
*requirements-docs.txt* installed:

::

    python tasks.py --task=task_make_docs

Search for **WARNING: Cannot resolve cyclic import** statement.