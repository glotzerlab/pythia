#!/usr/bin/env python

from setuptools import setup

with open('pythia/version.py') as version_file:
    exec(version_file.read())

setup(name='pythia',
      version=__version__,
      description='Machine learning fingerprints for particle environments',
      author='Matthew Spellings',
      author_email='mspells@umich.edu',
      classifiers=[
          'License :: OSI Approved :: BSD License'
      ],
      package_dir={'pythia': 'pythia'},
      packages=['pythia'],
      install_requires=['numpy', 'scipy', 'freud'],
      )
