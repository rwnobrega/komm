# Changelog

## v0.26.0 (2025-10-03)

### Breaking changes

- Changed the default constructor of `UniformQuantizer` to take `step` and `offset` instead of `input_range` and `choice`. Mid-tread and mid-riser uniform quantizers now have their own constructors. For example, instead of

  ```
  komm.UniformQuantizer(num_levels=4, input_range=(-1.0, 1.0), choice='mid-riser')
  ```

  use

  ```
  komm.UniformQuantizer.mid_riser(num_levels=4, step=0.5)
  ```

- Renamed method `entropy` to `entropy_rate` in `DiscreteMemorylessSource`.

### Added

- Implemented [finite-state Markov chains](https://komm.dev/ref/MarkovChain).

- Implemented [Lempel–Ziv 77](https://komm.dev/ref/LempelZiv77Code) lossless compression code — thanks [RhenzoHideki](https://github.com/RhenzoHideki/).

- Added token methods to Lempel–Ziv 78 code.

## v0.25.0 (2025-08-05)

### Breaking changes

- Removed modulation classes in favor of decoupled [constellation](https://komm.dev/ref/Constellation) and [labeling](https://komm.dev/ref/Labeling) classes. For example, instead of

  ```
  modulation = komm.QAModulation(16, labeling="reflected_2d")
  ```

  use

  ```
  constellation = komm.QAMConstellation(16)  # 16 symbols
  labeling = komm.ReflectedRectangularLabeling(4)  # 4 bits per symbol
  ```

  - To recover old `modulate()` behavior: Instead of

    ```
    symbols = modulation.modulate(bits)
    ```

    use

    ```
    indices = labeling.bits_to_indices(bits)
    symbols = constellation.indices_to_symbols(indices)
    ```

  - To recover old `demodulate_hard()` behavior: Instead of

    ```
    bits_hat = modulation.demodulate_hard(received)
    ```

    use

    ```
    indices_hat = constellation.closest_indices(received)
    bits_hat = labeling.indices_to_bits(indices_hat)
    ```

  - To recover old `demodulate_soft()` behavior: Instead of

    ```
    l_values = modulation.demodulate_soft(received, snr=5.0)
    ```

    use

    ```
    posteriors = constellation.posteriors(received, snr=5.0)
    l_values = labeling.marginalize(posteriors)
    ```

- The phase offset of constellations are now measured in [turns](<https://en.wikipedia.org/wiki/Turn_(angle)>) instead of radians.

- Renamed `__call__` methods to `emit`, in sources; `transmit`, in channels; and `decode`, in decoders.

### Added

- Added flag `bit_order` to [`bits_to_int`](https://komm.dev/ref/bits_to_int) and [`int_to_bits`](https://komm.dev/ref/int_to_bits) helper functions.

## v0.24.0 (2025-06-18)

### Breaking changes

- Converted `generator_matrix` from method to cached property in convolutional codes.

- Removed `TransmitFilter` class and moved some functionality to pulse classes. Pulse formatting should now be performed directly with NumPy's `convolve`, the new pulse `taps` method, and the new `sampling_rate_expand` helper function:

  ```
  pulse = komm.RaisedCosinePulse(rolloff=rolloff)
  info = [3, 1, -1, 1, -3, -1, 1, -3, 3, 1]
  x = komm.sampling_rate_expand(info, factor=sps)
  p = pulse.taps(samples_per_symbol=sps, span=(-8, 8))
  y = np.convolve(x, p)
  t = np.arange(y.size) / sps
  t -= 8  # Compensate for delay if desired
  ```

### Added

- Added sampling rate [compression](https://komm.dev/ref/sampling_rate_compress) and [expansion](https://komm.dev/ref/sampling_rate_expand) helper functions.
- Added [Fourier transform](https://komm.dev/ref/fourier_transform) helper function.

### Fixed

- Restored methods `constraint_lengths`, `overall_constraint_length`, and `memory_order` (according to [JZ15] definitions) in convolutional codes.

## v0.23.0 (2025-05-30)

### Breaking changes

- The property `transfer_function_matrix` of convolutional codes is now the method `generator_matrix()`.
- The property `overall_constraint_length` was renamed to `degree`.

### Added

- Implemented [`LowRateConvolutionalCode`](https://komm.dev/ref/LowRateConvolutionalCode) and [`HighRateConvolutionalCode`](https://komm.dev/ref/LowRateConvolutionalCode) classes.
- Added `is_catastrophic` method to convolutional codes.

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
