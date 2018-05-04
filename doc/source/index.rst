==================================
Welcome to pythia's documentation!
==================================

Pythia is a library to generate numerical descriptions of particle
systems. Most methods rely heavily on `freud
<https://bitbucket.org/glotzer/freud>`_ for efficient neighbor search
and other accelerated calculations.

Installation
============

Pythia is available on PyPI as `pythia-learn`::

  $ pip install pythia-learn

You can install pythia from source like this::

   $ git clone https://bitbucket.org/glotzer/pythia.git
   $ # now install
   $ cd pythia && python setup.py install --user

.. note::

   If using conda or a virtualenv, the `--user` argument in the pip
   command above is unnecessary.

Citation
========

In addition to the citations referenced in the docstring of each
function, we encourage users to cite the pythia project itself.

Documentation
=============

The documentation is available as standard sphinx documentation::

  $ cd doc
  $ make html

Automatically-built documentation is available at
https://pythia-learn.readthedocs.io .

Contents:
=========

.. toctree::
   :maxdepth: 2

   bonds
   spherical_harmonics
   voronoi

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
