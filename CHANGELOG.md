# Changelog

> [!NOTE]
> Changelog started with version v0.10.0.

## v0.12.0 (2024-12-05)

### Added

- Implemented [Marcum Q-function](https://komm.dev/ref/marcum_q).

### Breaking changes

- Renamed `int2binlist` to `int_to_bits`, `binlist2int` to `bits_to_int`, `qfunc` to `gaussian_q`, `qfuncinv` to `gaussian_q_inv`, `acorr` to `autocorrelation`, and `cyclic_acorr` to `cyclic_autocorrelation`, for consistency with other functions.
- Removed `pack` and `unpack` functions from `komm` module. Instead of `komm.pack(arr, width)`, use `komm.bits_to_int(arr.reshape(-1, width))`; and instead of `komm.unpack(arr, width)`, use `komm.int_to_bits(arr, width).ravel()`.

## v0.11.0 (2024-12-01)

### Breaking changes

- Converted property `finite_state_machine` of `ConvolutionalCode` to a method.
- Removed properties `state_matrix`, `control_matrix`, `observation_matrix`, `transition_matrix` from `ConvolutionalCode`, and replaced them with the method `state_space_representation()`. The new usage is `state_matrix, control_matrix, observation_matrix, transition_matrix = convolutional_code.state_space_representation()`.

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
  - Removed `'unquant'` choice (use `input_range=(0.0, input_peak)` and `choice="mid-tread"` instead).
- Converted the classes `BinaryErasureChannel`, `BinarySymmetricChannel`, `DiscreteMemorylessChannel`, `AWGNChannel`, `FixedToVariableEncoder`, `FixedToVariableDecoder`, `VariableToFixedEncoder`, `VariableToFixedDecoder`, and `DiscreteMemorylessSource` from mutable to immutable.
- Removed `RationalPolynomial` and `RationalPolynomialFraction` classes.
- Refactored _algebra_ and _pulse_ modules. See documentation for new usage.
- Adjusted string literals in `BlockDecoder` to kebab-case. For example, `method="exhaustive_search_hard"` should become `method="exhaustive-search-hard"`
