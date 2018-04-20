from importlib import import_module


def assert_installed(name):
    try:
        return import_module(name)
    except ImportError as error:
        raise ImportError("{} required for requested functionality.".format(name))
