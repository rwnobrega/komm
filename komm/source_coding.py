"""
Source coding
=============

Source coding...

Lossless coding
---------------

    SymbolCode
    HuffmanCode
"""

import numpy as np

__all__ = ['SymbolCode', 'HuffmanCode']


class SymbolCode:
    """
    Binary symbol code.

    A *symbol code* .... Variable length

    So far, only *instantaneous* (or *prefix-free*) codes are supported.
    """
    def __init__(self, dictionary):
        self._dictionary = {key: tuple(value) for (key, value) in dictionary.items()}
        self._reverse_dictionary = {tuple(value): key for (key, value) in dictionary.items()}

    @property
    def dictionary(self):
        """
        The dictionary of the symbol code.
        """
        return self._dictionary

    def encode(self, symbols):
        """
        Encode symbols to bits.
        """
        return np.concatenate([self._dictionary[symbol] for symbol in symbols])

    def average_length(self, probabilities):
        return np.dot([len(bit_sequence) for bit_sequence in self._dictionary.values()],
                      probabilities) / np.sum(probabilities)

    def decode(self, bits):
        """
        Decode bits to symbols.
        """
        symbols = []
        i = 0
        while i < len(bits):
            j = 1
            while True:
                bit_sequence = tuple(bits[i : i + j])
                if bit_sequence in self._reverse_dictionary:
                    break
                else:
                    j += 1
            symbols.append(self._reverse_dictionary[bit_sequence])
            i += j
        return np.array(symbols)

    def __repr__(self):
        args = 'dictionary={}'.format(self._dictionary)
        return '{}({})'.format(self.__class__.__name__, args)


class HuffmanCode(SymbolCode):
    """
    Huffman code.
    """
    def __init__(self, probabilities):
        pass


class TunstallCode:
    """
    Tunstall code
    """
    pass
