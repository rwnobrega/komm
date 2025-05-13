# Changelog

## v0.22.0 (2025-05-13)

### Breaking changes

- Renamed `FiniteStateMachine` to `MealyMachine` and its corresponding constructor parameter `next_states` to `transitions`.

- Removed `ConvolutionalStreamEncoder` and add encoding methods to `ConvolutionalCode`. Instead of

  ```
  code = komm.ConvolutionalCode(...)
  encoder = komm.ConvolutionalStreamEncoder(code, initial_state)
  output = encoder(input)
  final_state = encoder.state
  ```

  use

  ```
  code = komm.ConvolutionalCode(...)
  output, final_state = code.encode_with_state(input, initial_state)
  ```

  or, to encode starting from the zero state, use

  ```
  code = komm.ConvolutionalCode(...)
  output = code.encode(input)
  ```

  Note that in the latter case the final state is not returned.

### Added

- Implemented [Moore machine](https://komm.dev/ref/MooreMachine).
- Implemented `free_distance` method for convolutional codes.

## v0.21.0 (2025-05-04)

### Breaking changes

- Modified BCJR decoder to accept L-values instead of direct channel output. As a consequence, the parameter `snr` of `BCJRDecoder` was removed. Instead of

  ```
  >>> decoder = komm.BCJRDecoder(code, snr=snr)
  >>> lo = decoder(r)
  ```

  use, for example,

  ```
  >>> decoder = komm.BCJRDecoder(code)
  >>> bpsk = komm.PSKModulation(2)
  >>> li = bpsk.demodulate_soft(r, snr=snr)
  >>> lo = decoder(li)
  ```

- Replaced `time()` convenience method with `axes()` in `TransmitFilter` to return both time and frequency axes. Instead of `ts = tx_filter.time()`, use `ts, fs = tx_filter.axes()`.

- Changed default value of rollof of root-raised-cosine pulse (from `0.0` to `1.0`) for consistency with raised-cosine pulse.

### Added

- Implemented [Polar codes](https://komm.dev/ref/PolarCode) and its [successive cancellation decoder](https://komm.dev/ref/SCDecoder).
- Added [`boxplus`](https://komm.dev/ref/boxplus) and [`binary_entropy_inv`](https://komm.dev/ref/binary_entropy_inv) utility functions.
- Implemented `autocorrelation` and `energy_density_spectrum` methods for pulses.

### Fixed

- Adjusted spectrum of pulses to return correct phases.

## v0.20.1 (2025-04-04)

### Added

- Implemented `mean_squared_error` method for quantizers.

### Fixed

- Fixed `BCJRDecoder` for the case of $k > 1$.

## v0.20.0 (2025-03-27)

### Breaking changes

- Renamed `__call__` method to `quantize`, and added `digitize` method in quantizers.

## v0.19.0 (2025-03-13)

### Breaking changes

- Adopt MSB-first instead of LSB-first for the labeling of modulation schemes.

### Added

- Implemented [unary code](https://komm.dev/ref/UnaryCode) and [Fibonacci code](https://komm.dev/ref/FibonacciCode).

## v0.18.0 (2025-03-10)

### Breaking changes

- Adopt MSB-first instead of LSB-first for `LempelZiv78` and `LempelZivWelch` classes.
- Hide `inv_enc_mapping` from `FixedToVariableCode`, and `inv_dec_mapping` from `VariableToFixedCode` classes.

## v0.17.0 (2025-02-26)

### Added

- Implemented [Lempel–Ziv 78](https://komm.dev/ref/LempelZiv78Code) and [Lempel–Ziv–Welch](https://komm.dev/ref/LempelZivWelchCode) lossless data compression algorithms.

## v0.16.2 (2025-02-18)

### Fixed

- Fixed regression in block decoders when processing input with errors beyond their error correction capability.

## v0.16.1 (2025-01-04)

### Added

- Added size property to lossless source coding codes.

## v0.16.0 (2025-01-03)

### Added

- Implemented [Shannon](https://komm.dev/ref/ShannonCode) and [Fano](https://komm.dev/ref/FanoCode) codes.
- Implemented method to compute the [Kraft parameter](https://komm.dev/ref/FixedToVariableCode#kraft_parameter) of a fixed-to-variable code.

## v0.15.1 (2024-12-31)

### Fixed

- Fixed `parse_prefix_free` when `allow_incomplete=True`.

## v0.15.0 (2024-12-30)

### Fixed

- Fixed Sardinas–Patterson algorithm.
- Encoding and decoding of [variable-to-fixed codes](https://komm.dev/ref/VariableToFixedCode) now require the code to be [fully covering](https://komm.dev/ref/VariableToFixedCode#is_fully_covering).

## v0.14.0 (2024-12-19)

### Breaking changes

- Removed `primitive_element` method from `FiniteField` class, since the implementation was incorrect. The method was assuming that the modulus was always primitive (which may not be true).

### Added

- Added polynomial [irreducibility](https://komm.dev/ref/BinaryPolynomial#is_irreducible) and [primitivity](https://komm.dev/ref/BinaryPolynomial#is_primitive) tests for binary polynomials.

- Restored sequence support in block code and decoders methods.

  - Block code methods `encode`, `inverse_encode`, `check`, and block decoders `__call__` now accept sequences spanning multiple blocks (as before v0.13.0) as well as multidimensional arrays.

- Added support for multidimensional input in modulation, demodulation, and channel methods.

## v0.13.0 (2024-12-12)

### Breaking changes

- Removed `BlockEncoder` and `BlockDecoder` classes in favor of direct methods and specialized decoder classes.

  - Block code methods are now called directly: `code.encode(u)`, `code.inverse_encode(v)`, and `code.check(r)` (previously `enc_mapping`, `inv_enc_mapping`, and `chk_mapping`). These methods are now vetorized (i.e., support input arrays of any shape). For example, for a code with dimension $k = 2$, instead of `encoder = BlockEncoder(code); encoder([0, 1, 0, 1])`, use `code.encode([[0, 1], [0, 1]])`.
  - Decoder methods are now individual classes: `BCJRDecoder`, `BerlekampDecoder`, `ExhaustiveSearchDecoder`, `ReedDecoder`, `SyndromeTableDecoder`, `ViterbiDecoder`, and `WagnerDecoder`. For example, instead of `decoder = komm.BlockDecoder(code, method="exhaustive-search-hard")`, use `decoder = komm.ExhaustiveSearchDecoder(code, input_type="hard")`. The decoder `__call__` are now vectorized (i.e., support input arrays of any shape). For example, for a code with length $n = 3$, instead of `decoder([0, 1, 0, 1, 1, 0])`, use `decoder([[0, 1, 0], [1, 1, 0]])`.
  - The decoders `majority-logic-repetition-code` and `meggitt` were removed for now.

- Renamed `ConvolutionalStreamDecoder` to `ViterbiStreamDecoder`.

- Merged lossless source coding encoders/decoders into code classes.

  - Removed `FixedToVariableEncoder`, `FixedToVariableDecoder`, `VariableToFixedEncoder`, `VariableToFixedDecoder`. For example, instead of `encoder = FixedToVariableEncoder(code); output = encoder(input)`, use `output = code.encode(input)`.

## v0.12.0 (2024-12-05)

### Added

- Implemented [Marcum Q-function](https://komm.dev/ref/marcum_q).

### Breaking changes

- Renamed `int2binlist` to `int_to_bits`, `binlist2int` to `bits_to_int`, `qfunc` to `gaussian_q`, `qfuncinv` to `gaussian_q_inv`, `acorr` to `autocorrelation`, and `cyclic_acorr` to `cyclic_autocorrelation`, for consistency with other functions.
- Removed `pack` and `unpack` functions from `komm` module. Instead of `komm.pack(arr, width)`, use `komm.bits_to_int(arr.reshape(-1, width))`; and instead of `komm.unpack(arr, width)`, use `komm.int_to_bits(arr, width).ravel()`.

## v0.11.0 (2024-12-01)

### Breaking changes

- Converted property `finite_state_machine` of `ConvolutionalCode` to a method.
- Removed properties `state_matrix`, `control_matrix`, `observation_matrix`, `transition_matrix` from `ConvolutionalCode`, and replaced them with the method `state_space_representation`. The new usage is `state_matrix, control_matrix, observation_matrix, transition_matrix = convolutional_code.state_space_representation()`.

## v0.10.0 (2024-11-29)

### Added

- Implemented [relative entropy](https://komm.dev/ref/relative_entropy) (KL divergence) function.
- Implemented [Slepian array](https://komm.dev/ref/SlepianArray).
- Implemented [Lloyd-Max quantizer](https://komm.dev/ref/LloydMaxQuantizer).
- Implemented [Z-Channel](https://komm.dev/ref/ZChannel).
- Implemented [lexicodes](https://komm.dev/ref/Lexicode).
- Added progress bar (via `tqdm`) to potential slow methods.

### Breaking changes

- Converted cached properties to cached methods.
  - In `BlockCode`: `codewords`, `codeword_weight_distribution`, `minimum_distance`, `coset_leaders`, `coset_leader_weight_distribution`, `packing_radius`, and `covering_radius`.
  - In `ReedMullerCode`: `reed_partitions`.
- In `UniformQuantizer`:
  - Replaced `input_peak` with `input_range`.
  - Removed `"unquant"` choice (use `input_range=(0.0, input_peak)` and `choice="mid-tread"` instead).
- Converted the classes `BinaryErasureChannel`, `BinarySymmetricChannel`, `DiscreteMemorylessChannel`, `AWGNChannel`, `FixedToVariableEncoder`, `FixedToVariableDecoder`, `VariableToFixedEncoder`, `VariableToFixedDecoder`, and `DiscreteMemorylessSource` from mutable to immutable.
- Removed `RationalPolynomial` and `RationalPolynomialFraction` classes.
- Refactored _algebra_ and _pulse_ modules. See documentation for new usage.
- Adjusted string literals in `BlockDecoder` to kebab-case. For example, `method="exhaustive_search_hard"` should become `method="exhaustive-search-hard"`

> [!NOTE]
> Changelog started with version v0.10.0.
