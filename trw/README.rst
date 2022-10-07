Torch Research Workflows
************************

.. image:: https://dev.azure.com/civodlu/trw/_apis/build/status/civodlu.trw?branchName=master
    :target: https://dev.azure.com/civodlu/trw/_build
   

   
.. image:: https://readthedocs.org/projects/trw/badge/?version=latest
	:target: https://trw.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status


.. image:: https://coveralls.io/repos/github/civodlu/trw/badge.svg?branch=master
	:target: https://coveralls.io/github/civodlu/trw?branch=master

Purpose
=======

The aim of this library is to simplify the process of building, optimizing, testing and debugging
deep learning models using PyTorch as well as providing implementations of some of the latest
research papers. Extensibility is kept in mind so that it is easy to customize the framework for
your particular needs.

Some key features of the framework:

* Easy to use, flexible and extensible API to build simple & complex models 
* Model debugging (e.g., activation statistics of each layer, gradient norm for each layer, embedding visualization)
* Model understanding and result analysis (e.g., attention maps, confusion matrix, ROC curves, model comparisons, errors)
* Support hyper-parameter optimization (random search, hyperband) and analysis
* Architecture learning (DARTS & evolutionary algorithms)
* Keep track of the results for retrospective analysis and model selection

Requirements
============

* Linux/Windows
* Python >= 3.6
* PyTorch >= 1.0

Installation / Usage
====================

To install use pip:

    $ pip install trw


Or clone the repo:

    $ git clone https://github.com/civodlu/trw.git
    
    $ python setup.py install
    
Documentation
=============

The documentation can be found at ReadTheDocs_.

.. _ReadTheDocs: https://trw.readthedocs.io/en/latest/
	
