---
hide: navigation
---

# Library reference

## Algebra

- [BinaryPolynomial](BinaryPolynomial). Binary polynomial.
- [BinaryPolynomialFraction](BinaryPolynomialFraction). Binary polynomial fraction.
- [FiniteBifield](FiniteBifield). Finite field with binary characteristic.
- [RationalPolynomial](RationalPolynomial). Rational polynomial.
- [RationalPolynomialFraction](RationalPolynomialFraction). Rational polynomial fraction.

## Channels

- [AWGNChannel](AWGNChannel). Additive white Gaussian noise (AWGN) channel.
- [DiscreteMemorylessChannel](DiscreteMemorylessChannel). Discrete memoryless channel (DMC).
- [BinarySymmetricChannel](BinarySymmetricChannel). Binary symmetric channel (BSC).
- [BinaryErasureChannel](BinaryErasureChannel). Binary erasure channel (BEC).

## Error control

### Block coding

- [BlockCode](BlockCode). General binary linear block code.
- [HammingCode](HammingCode). Hamming code.
- [SimplexCode](SimplexCode). Simplex (maximum-length) code.
- [GolayCode](GolayCode). Binary Golay code.
- [RepetitionCode](RepetitionCode). Repetition code.
- [SingleParityCheckCode](SingleParityCheckCode). Single parity check code.
- [CordaroWagnerCode](CordaroWagnerCode). Cordaro--Wagner code.
- [ReedMullerCode](ReedMullerCode). Reed--Muller code.
- [CyclicCode](CyclicCode). General binary cyclic code.
- [BCHCode](BCHCode). Bose--Chaudhuri--Hocquenghem (BCH) code.

### Convolutional coding

- [ConvolutionalCode](ConvolutionalCode). Binary convolutional code.
- [ConvolutionalStreamEncoder](ConvolutionalStreamEncoder). Convolutional stream encoder.
- [ConvolutionalStreamDecoder](ConvolutionalStreamDecoder). Convolutional stream decoder using Viterbi algorithm.
- [TerminatedConvolutionalCode](TerminatedConvolutionalCode). Terminated convolutional code.

## Finite-state machines

- [FiniteStateMachine](FiniteStateMachine). Finite-state machine (Mealy machine).

## Modulation

### Real modulation schemes

- [RealModulation](RealModulation). General real modulation scheme.
- [PAModulation](PAModulation). Pulse-amplitude modulation (PAM).

### Complex modulation schemes

- [ComplexModulation](ComplexModulation). General complex modulation scheme.
- [ASKModulation](ASKModulation). Amplitude-shift keying (ASK) modulation.
- [PSKModulation](PSKModulation). Phase-shift keying (PSK) modulation.
- [APSKModulation](APSKModulation). Amplitude- and phase-shift keying (APSK) modulation.
- [QAModulation](QAModulation). Quadrature-amplitude modulation (QAM).

## Pulse formatting

### Pulses

- [RectangularPulse](RectangularPulse). Rectangular pulse.
- [ManchesterPulse](ManchesterPulse). Manchester pulse.
- [SincPulse](SincPulse). Sinc pulse.
- [RaisedCosinePulse](RaisedCosinePulse). Raised cosine pulse.
- [RootRaisedCosinePulse](RootRaisedCosinePulse). Root raised cosine pulse.
- [GaussianPulse](GaussianPulse). Gaussian pulse.

### Filtering

- [TransmitFilter](TransmitFilter). Transmit filter.
- [ReceiveFilter](ReceiveFilter). Receive filter [Not implemented yet].

## Quantization

- [ScalarQuantizer](ScalarQuantizer). General scalar quantizer.
- [LloydMaxQuantizer](LloydMaxQuantizer). Lloyd--Max scalar quantizer [Not implemented yet].
- [UniformQuantizer](UniformQuantizer). Uniform scalar quantizer.

## Sequences

### Binary sequences

- [BinarySequence](BinarySequence). General binary sequence.
- [BarkerSequence](BarkerSequence). Barker sequence.
- [WalshHadamardSequence](WalshHadamardSequence). Walsh--Hadamard sequence.
- [LFSRSequence](LFSRSequence). Linear-feedback shift register (LFSR) sequence.
- [GoldSequence](GoldSequence). Gold sequence [Not implemented yet].
- [KasamiSequence](KasamiSequence). Kasami sequence [Not implemented yet].

### Other sequences

- [ZadoffChuSequence](ZadoffChuSequence). Zadoff–Chu sequence [Not implemented yet].

## Source coding

### Sources

- [DiscreteMemorylessSource](DiscreteMemorylessSource). Discrete memoryless source (DMS).

### Lossless coding

- [FixedToVariableCode](FixedToVariableCode). Binary, prefix-free, fixed-to-variable length code.
- [HuffmanCode](HuffmanCode). Huffman code.
- [VariableToFixedCode](VariableToFixedCode). Binary, prefix-free, variable-to-fixed length code.
- [TunstallCode](TunstallCode). Tunstall code.

## Utilities

- [binlist2int](binlist2int). Converts a bit array to its integer representation.
- [int2binlist](int2binlist). Converts an integer to its bit array representation.
- [pack](pack). Packs a given integer array.
- [unpack](unpack). Unpacks a given bit array.
- [qfunc](qfunc). Computes the Gaussian Q-function.
- [qfuncinv](qfuncinv). Computes the inverse Gaussian Q-function.
- [entropy](entropy). Computes the entropy of a random variable with a given :term:`pmf`.

