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

  $ pip install pythia-learn freud-analysis

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

Usage
=====

In general, data types follow the `hoomd-blue schema
<http://hoomd-blue.readthedocs.io/en/stable/box.html>`_:

- Positions are an Nx3 array of particle coordinates, with `(0, 0, 0)` being the center of the box
- Boxes are specified as an object with `Lx`, `Ly`, `Lz`, `xy`, `xz`, and `yz` elements
- Orientations are specified as orientation quaternions: an Nx4 array of `(r, i, j, k)` elements

Examples
========

Example notebooks are available in the `examples` directory:

- `Unsupervised learning <https://bitbucket.org/glotzer/pythia/src/master/examples/Unsupervised%20Learning.ipynb?viewer=nbviewer>`_
- `Supervised learning <https://bitbucket.org/glotzer/pythia/src/master/examples/Supervised%20Learning.ipynb?viewer=nbviewer>`_
- `Steinhardt and Pythia order parameter comparison (FCC and HCP) <https://bitbucket.org/glotzer/pythia/src/master/examples/Steinhardt FCC HCP comparison.ipynb?viewer=nbviewer>`_

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
