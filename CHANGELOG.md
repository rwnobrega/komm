# Changelog

> [!NOTE]
> Changelog started with version v0.10.0.

## v0.10.0

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
