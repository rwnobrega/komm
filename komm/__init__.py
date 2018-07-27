__version__ = '0.6.0'

from ._algebra import *
from ._channels import *
from ._error_control_block import *
from ._error_control_convolutional import *
from ._finite_state_machine import *
from ._modulation import *
from ._pulses import *
from ._quantization import *
from ._sequences import *
from ._source_coding import *
from ._util import *

import inspect as _inspect
import sys as _sys

for _, _cls in _inspect.getmembers(_sys.modules[__name__], _inspect.isclass):
    if hasattr(_cls, '_process_docstring'):
        _cls._process_docstring()
