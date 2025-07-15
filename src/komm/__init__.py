from importlib import metadata as _metadata

__version__ = _metadata.version("komm")


# fmt: off
# isort: off

# Algebra
from ._algebra.BinaryPolynomial import BinaryPolynomial
from ._algebra.BinaryPolynomialFraction import BinaryPolynomialFraction
from ._algebra.FiniteBifield import FiniteBifield
from ._algebra.Integers import Integer

# Channels
from ._channels.AWGNChannel import AWGNChannel
from ._channels.BinaryErasureChannel import BinaryErasureChannel
from ._channels.BinarySymmetricChannel import BinarySymmetricChannel
from ._channels.DiscreteMemorylessChannel import DiscreteMemorylessChannel
from ._channels.ZChannel import ZChannel

# Constellations
from ._constellations.ASKConstellation import ASKConstellation
from ._constellations.APSKConstellation import APSKConstellation
from ._constellations.Constellation import Constellation
from ._constellations.PAMConstellation import PAMConstellation
from ._constellations.PSKConstellation import PSKConstellation
from ._constellations.QAMConstellation import QAMConstellation

# Error control - block codes
from ._error_control_block.BCHCode import BCHCode
from ._error_control_block.BlockCode import BlockCode
from ._error_control_block.CordaroWagnerCode import CordaroWagnerCode
from ._error_control_block.CyclicCode import CyclicCode
from ._error_control_block.GolayCode import GolayCode
from ._error_control_block.HammingCode import HammingCode
from ._error_control_block.Lexicode import Lexicode
from ._error_control_block.PolarCode import PolarCode
from ._error_control_block.ReedMullerCode import ReedMullerCode
from ._error_control_block.RepetitionCode import RepetitionCode
from ._error_control_block.SimplexCode import SimplexCode
from ._error_control_block.SingleParityCheckCode import SingleParityCheckCode
from ._error_control_block.SlepianArray import SlepianArray
from ._error_control_block.SystematicBlockCode import SystematicBlockCode

# Error control - checksum
from ._error_control_checksum.CyclicRedundancyCheck import CyclicRedundancyCheck

# Error control - convolutional
from ._error_control_convolutional.ConvolutionalCode import ConvolutionalCode
from ._error_control_convolutional.HighRateConvolutionalCode import HighRateConvolutionalCode
from ._error_control_convolutional.LowRateConvolutionalCode import LowRateConvolutionalCode
from ._error_control_convolutional.TerminatedConvolutionalCode import TerminatedConvolutionalCode
from ._error_control_convolutional.ViterbiStreamDecoder import ViterbiStreamDecoder

# Error control - decoders
from ._error_control_decoders.BCJRDecoder import BCJRDecoder
from ._error_control_decoders.BerlekampDecoder import BerlekampDecoder
from ._error_control_decoders.ExhaustiveSearchDecoder import ExhaustiveSearchDecoder
from ._error_control_decoders.ReedDecoder import ReedDecoder
from ._error_control_decoders.SCDecoder import SCDecoder
from ._error_control_decoders.SyndromeTableDecoder import SyndromeTableDecoder
from ._error_control_decoders.ViterbiDecoder import ViterbiDecoder
from ._error_control_decoders.WagnerDecoder import WagnerDecoder

# Finite state machines
from ._finite_state_machine.MealyMachine import MealyMachine
from ._finite_state_machine.MooreMachine import MooreMachine

# Integer coding
from ._integer_coding.FibonacciCode import FibonacciCode
from ._integer_coding.UnaryCode import UnaryCode

# Labelings
from ._labelings.Labeling import Labeling
from ._labelings.NaturalLabeling import NaturalLabeling
from ._labelings.ProductLabeling import ProductLabeling
from ._labelings.ReflectedLabeling import ReflectedLabeling
from ._labelings.ReflectedRectangularLabeling import ReflectedRectangularLabeling

# Lossless coding
from ._lossless_coding.FanoCode import FanoCode
from ._lossless_coding.FixedToVariableCode import FixedToVariableCode
from ._lossless_coding.HuffmanCode import HuffmanCode
from ._lossless_coding.LempelZiv78Code import LempelZiv78Code
from ._lossless_coding.LempelZivWelchCode import LempelZivWelchCode
from ._lossless_coding.ShannonCode import ShannonCode
from ._lossless_coding.TunstallCode import TunstallCode
from ._lossless_coding.VariableToFixedCode import VariableToFixedCode

# Pulses
from ._pulses.GaussianPulse import GaussianPulse
from ._pulses.ManchesterPulse import ManchesterPulse
from ._pulses.Pulse import Pulse
from ._pulses.RaisedCosinePulse import RaisedCosinePulse
from ._pulses.RectangularPulse import RectangularPulse
from ._pulses.RootRaisedCosinePulse import RootRaisedCosinePulse
from ._pulses.SincPulse import SincPulse

# Quantization
from ._quantization.LloydMaxQuantizer import LloydMaxQuantizer
from ._quantization.ScalarQuantizer import ScalarQuantizer
from ._quantization.UniformQuantizer import UniformQuantizer

# Sequences - binary
from ._sequences_binary.BarkerSequence import BarkerSequence
from ._sequences_binary.BinarySequence import BinarySequence
from ._sequences_binary.GoldSequence import GoldSequence
from ._sequences_binary.KasamiSequence import KasamiSequence
from ._sequences_binary.LFSRSequence import LFSRSequence
from ._sequences_binary.WalshHadamardSequence import WalshHadamardSequence

# Sequences - complex
from ._sequences_complex.ComplexSequence import ComplexSequence
from ._sequences_complex.ZadoffChuSequence import ZadoffChuSequence

# Sources
from ._sources.DiscreteMemorylessSource import DiscreteMemorylessSource

# Util
from ._util import global_rng
from ._util.bit_operations import bits_to_int, int_to_bits
from ._util.correlation import autocorrelation, cyclic_autocorrelation
from ._util.information_theory import binary_entropy, binary_entropy_inv, entropy, relative_entropy
from ._util.signal_processing import fourier_transform, sampling_rate_compress, sampling_rate_expand
from ._util.special_functions import boxplus, gaussian_q, gaussian_q_inv, marcum_q

# isort: on
# fmt: on


__all__ = [
    # _algebra
    "BinaryPolynomial",
    "BinaryPolynomialFraction",
    "FiniteBifield",
    "Integer",
    # _channels
    "AWGNChannel",
    "BinaryErasureChannel",
    "BinarySymmetricChannel",
    "DiscreteMemorylessChannel",
    "ZChannel",
    # _constellations
    "APSKConstellation",
    "ASKConstellation",
    "Constellation",
    "PAMConstellation",
    "PSKConstellation",
    "QAMConstellation",
    # _error_control_block
    "BCHCode",
    "BlockCode",
    "CordaroWagnerCode",
    "CyclicCode",
    "GolayCode",
    "HammingCode",
    "Lexicode",
    "PolarCode",
    "ReedMullerCode",
    "RepetitionCode",
    "SimplexCode",
    "SingleParityCheckCode",
    "SlepianArray",
    "SystematicBlockCode",
    # _error_control_checksum
    "CyclicRedundancyCheck",
    # _error_control_convolutional
    "ConvolutionalCode",
    "HighRateConvolutionalCode",
    "LowRateConvolutionalCode",
    "TerminatedConvolutionalCode",
    "ViterbiStreamDecoder",
    # _error_control_decoders
    "BCJRDecoder",
    "BerlekampDecoder",
    "ExhaustiveSearchDecoder",
    "ReedDecoder",
    "SCDecoder",
    "SyndromeTableDecoder",
    "ViterbiDecoder",
    "WagnerDecoder",
    # _finite_state_machine
    "MealyMachine",
    "MooreMachine",
    # _integer_coding
    "FibonacciCode",
    "UnaryCode",
    # _labelings
    "Labeling",
    "NaturalLabeling",
    "ProductLabeling",
    "ReflectedLabeling",
    "ReflectedRectangularLabeling",
    # _lossless_coding
    "FanoCode",
    "FixedToVariableCode",
    "HuffmanCode",
    "LempelZiv78Code",
    "LempelZivWelchCode",
    "ShannonCode",
    "TunstallCode",
    "VariableToFixedCode",
    # _pulses
    "GaussianPulse",
    "ManchesterPulse",
    "Pulse",
    "RaisedCosinePulse",
    "RectangularPulse",
    "RootRaisedCosinePulse",
    "SincPulse",
    # _quantization
    "LloydMaxQuantizer",
    "ScalarQuantizer",
    "UniformQuantizer",
    # _sequences_binary
    "BarkerSequence",
    "BinarySequence",
    "GoldSequence",
    "KasamiSequence",
    "LFSRSequence",
    "WalshHadamardSequence",
    # _sequences_complex
    "ComplexSequence",
    "ZadoffChuSequence",
    # _sources
    "DiscreteMemorylessSource",
    # _util
    "global_rng",
    "bits_to_int",
    "int_to_bits",
    "autocorrelation",
    "cyclic_autocorrelation",
    "binary_entropy",
    "binary_entropy_inv",
    "entropy",
    "relative_entropy",
    "fourier_transform",
    "sampling_rate_compress",
    "sampling_rate_expand",
    "boxplus",
    "gaussian_q",
    "gaussian_q_inv",
    "marcum_q",
]
