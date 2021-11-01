"""
:mod:`tobler` --- A library for spatial interpolation
=================================================

"""
from . import area_weighted
from . import dasymetric
from . import model
from . import util
from . import pycno

from . import _version
__version__ = _version.get_versions()['version']
