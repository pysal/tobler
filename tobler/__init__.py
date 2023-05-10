"""
:mod:`tobler` --- A library for spatial interpolation
=================================================

"""
import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import area_weighted, dasymetric, model, pycno, util

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("tobler")
