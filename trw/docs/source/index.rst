.. sectnum::

======================================
Torch Research Workflows Documentation
======================================


Purpose
*******

The aim of this library is to simplify the process of building, optimizing, testing and debugging
deep learning models using PyTorch as well as providing implementations of some of the latest
research papers. Extensibility is kept in mind so that it is easy to customize the framework for
your particular needs.

Some key features of the framework:

* Easy to use, flexible and extensible API to build simple & complex models with multiple inputs, outputs and tasks
* Model debugging (e.g., activation statistics of each layer, gradient norm for each layer, embedding visualization)
* Model understanding and result analysis (e.g., attention maps, confusion matrix, ROC curves, model comparisons, errors)
* Support hyper-parameter optimization (random search, hyperband) and analysis
* Architecture learning (DARTS & evolutionary algorithms)
* Keep track of the results for retrospective analysis and model selection


Contents:
*********

.. contents::


.. toctree::
   :maxdepth: 1
   

.. include:: example.rst
.. include:: simple_layers.rst
.. include:: input_pipeline.rst
.. include:: unbalanced_data.rst



