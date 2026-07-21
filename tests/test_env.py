"""Environment smoke test.

Exists so the very first pytest run collects at least one test (pytest exits with
status 5 on an empty collection), and so a broken/incomplete env fails loudly here
rather than deep inside a data test.
"""

import sys


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_pinned_imports():
    import kymatio  # noqa: F401
    import numpy  # noqa: F401
    import openpyxl  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    import yaml  # noqa: F401
