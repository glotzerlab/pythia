==================================
Welcome to pythia's documentation!
==================================

Pythia is a library to generate numerical descriptions of particle
systems. Most methods rely heavily on `freud
<https://github.com/glotzerlab/freud>`_ for efficient neighbor search
and other accelerated calculations.

Installation
============

Pythia is available on PyPI as `pythia-learn`::

  $ pip install pythia-learn freud-analysis

You can install pythia from source like this::

   $ git clone https://github.com/glotzerlab/pythia.git
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

- `Unsupervised learning <https://github.com/glotzerlab/pythia/blob/master/examples/Unsupervised%20Learning.ipynb>`_
- `Supervised learning <https://github.com/glotzerlab/pythia/blob/master/examples/Supervised%20Learning.ipynb>`_
- `Steinhardt and Pythia order parameter comparison (FCC and HCP) <https://github.com/glotzerlab/pythia/blob/master/examples/Steinhardt%20FCC%20HCP%20comparison.ipynb>`_

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
