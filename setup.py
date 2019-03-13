#!/usr/bin/env python

from setuptools import setup

with open('pythia/version.py') as version_file:
    exec(version_file.read())

long_description_lines = []
with open('README.rst', 'r') as readme:
    for line in readme:
        if line.startswith('Contents'):
            break
        long_description_lines.append(line)
long_description = ''.join(long_description_lines)

setup(name='pythia-learn',
      author='Matthew Spellings',
      author_email='mspells@umich.edu',
      classifiers=[
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      description='Machine learning fingerprints for particle environments',
      install_requires=[
          'numpy',
          'scipy',
          'freud-analysis',
      ],
      license='BSD',
      long_description=long_description,
      package_dir={'pythia': 'pythia'},
      packages=['pythia'],
      project_urls={
          'Documentation': 'https://pythia-learn.readthedocs.io/',
          'Source': 'https://github.com/glotzerlab/pythia'
      },
      python_requires='>=3',
      version=__version__  # noqa F821
      )
