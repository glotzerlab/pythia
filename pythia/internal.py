import functools
from importlib import import_module
import inspect
import logging
import textwrap

logger = logging.getLogger(__name__)


def assert_installed(name):
    try:
        return import_module(name)
    except ImportError:
        raise ImportError("{} required for requested functionality.".format(name))


all_citations = {}

all_citations['kondor2007'] = """
@article{kondor2007,
        title = {A novel set of rotationally and translationally invariant features for images based on the non-commutative bispectrum},
        url = {http://arxiv.org/abs/cs/0701127},
        journal = {arXiv:cs/0701127},
        author = {Kondor, Risi},
        month = jan,
        year = {2007},
}
"""  # noqa E501

all_citations['freud2016'] = """
@misc{freud2016,
        title = {freud},
        url = {https://doi.org/10.5281/zenodo.166564},
        abstract = {First official open-source release, includes a zenodo DOI for citations.},
        author = {Harper, Eric S and Spellings, Matthew and Anderson, Joshua A and Glotzer, Sharon C},
        month = nov,
        year = {2016},
        doi = {10.5281/zenodo.166564},
}
"""  # noqa E501

all_citations['spellings2018'] = """
@article{spellings2018,
        title = {Machine learning for crystal identification and discovery},
        volume = {64},
        url = {https://dx.doi.org/10.1002/aic.16157},
        doi = {10.1002/aic.16157},
        number = {6},
        journal = {AIChE Journal},
        author = {Spellings, Matthew and Glotzer, Sharon C},
        year = {2018},
        pages = {2198--2206},
}
"""


def _cite(f, extra_doc):
    if f.__doc__ is None:
        f.__doc__ = ''

    f.__doc__ = inspect.cleandoc(f.__doc__) + extra_doc
    return f


def cite(*args):
    """Decorator for adding citation notes to docstrings"""

    extra_segments = ['\n\nThis function uses the following citations::\n']

    for arg in args:
        try:
            citation = all_citations[arg]
            extra_segments.append(textwrap.indent(citation, 4*' '))
        except KeyError:
            logger.warning('Unknown citation {}'.format(arg))

    extra_doc = ''.join(extra_segments)

    return functools.partial(_cite, extra_doc=extra_doc)
