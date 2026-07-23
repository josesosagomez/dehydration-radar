"""Environment smoke test.

Exists so the very first pytest run collects at least one test (pytest exits with
status 5 on an empty collection), and so a broken/incomplete env fails loudly here
rather than deep inside a data test.
"""

import sys

from packaging.version import Version


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_pinned_imports():
    import kymatio  # noqa: F401
    import numpy  # noqa: F401
    import openpyxl  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    import torch  # noqa: F401  (M4: WST numpy/torch cross-backend validation)
    import yaml  # noqa: F401


def test_scipy_pin_survives_torch():
    """torch (added at M4) must not drag scipy past the kymatio <1.17 ceiling.

    Compared on PARSED versions, never as strings: "1.9" >= "1.17" lexicographically
    even though 1.9 < 1.17 as versions — exactly the bug parsed comparison prevents.
    kymatio 0.3.0's scattering3d filter bank imports scipy.special.sph_harm, removed
    in scipy 1.17, so a bump here would break `from kymatio.numpy import Scattering1D`.
    """
    import scipy

    assert Version(scipy.__version__) < Version("1.17")
