# Changelog

> [!NOTE]
> Changelog started with version v0.10.0.

## v0.11.0 (2024-12-01)

### BREAKING CHANGE

- The property `finite_state_machine` of `ConvolutionalCode` was converted to method.
- The properties `state_matrix`, `control_matrix`, `observation_matrix`, and `transition_matrix` were removed from `ConvolutionalCode`. The new usage is `state_matrix, control_matrix, observation_matrix, transition_matrix = convolutional_code.state_space_representation()`.

### Fix

- improve error reporting in `BlockDecoder` and `TerminatedConvolutionalCode`

### Refactor

- convert `finite_state_machine` property of `ConvolutionalCode` into method
- replace state-space representation properties with a single method returning all matrices

### Perf

- add `cache` decorator to state space representation method
- add `cached_property` decorator to `_tail_projector` and `_zs_multiplier` methods in termination strategies

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
