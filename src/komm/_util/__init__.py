# Functions beginning with underscore:
# - Should not be used by the end user.
# - Should be as fast as possible.
# - May have assumptions on the input and on the output
#   (e.g., may assume the input is a list, or a numpy array, etc.).
#
# Functions without underscore:
# - Are available to the end user.
# - Should work when the input is a list, a numpy array, etc.
# - Should check the input whenever possible.
# - Should return a numpy array (instead of a list) whenever possible.

# TODO: Rename binlist2int and int2binlist to something better.
# TODO: Vectorize those functions (e.g., axis=1).


from .bit_operations import (
    _binlist2int,
    _int2binlist,
    binlist2int,
    int2binlist,
    pack,
    unpack,
)
from .correlation import acorr, cyclic_acorr
from .information_theory import binary_entropy, entropy
from .matrices import cartesian_product
from .special_functions import qfunc, qfuncinv

__all__ = [
    "acorr",
    "binary_entropy",
    "binlist2int",
    "cyclic_acorr",
    "entropy",
    "int2binlist",
    "pack",
    "qfunc",
    "qfuncinv",
    "unpack",
]
