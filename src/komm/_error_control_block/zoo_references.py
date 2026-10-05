"""
Mappings between Komm error correction codes and Error Correction Zoo codes.

The Error Correction Zoo (https://errorcorrectionzoo.org/) is a comprehensive
database of error correcting codes. This module provides mappings to link Komm
codes to their corresponding Zoo entries using permanent code IDs.
"""

# Mapping of Komm code class names to Error Correction Zoo code IDs
ZOO_CODE_IDS = {
    "HammingCode": "hamming",
    "SimplexCode": "simplex",
    "RepetitionCode": "repetition",
    "SingleParityCheckCode": "single_parity_check",
    "GolayCode": "golay",
    "BCHCode": "bch",
    "ReedSolomonCode": "reed_solomon",
    "ReedMullerCode": "reed_muller",
    "PolarCode": "polar",
    "CordaroWagnerCode": "cordaro_wagner",
    "Lexicode": "lexicode",
    "CyclicCode": "cyclic",
}


def get_zoo_url(code_class_name: str) -> str:
    """
    Get the Error Correction Zoo URL for a given code class.

    Parameters:
        code_class_name: The name of the Komm code class (e.g., 'HammingCode')

    Returns:
        The permanent URL to the code's page on the Error Correction Zoo,
        or None if the code is not yet mapped.

    Examples:
        >>> get_zoo_url('HammingCode')
        'https://errorcorrectionzoo.org/c/hamming'
        >>> get_zoo_url('GolayCode')
        'https://errorcorrectionzoo.org/c/golay'
    """
    zoo_id = ZOO_CODE_IDS.get(code_class_name)
    if zoo_id is None:
        return None
    return f"https://errorcorrectionzoo.org/c/{zoo_id}"
