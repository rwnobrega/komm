# QR code format information

The _format information_ of a [QR code](https://en.wikipedia.org/wiki/QR_code) is a 15-bit word that gives the _error correction level_ and the _data mask pattern_ of the symbol. Its 5 information bits are the 2 bits of the error correction level (`01` for L, `00` for M, `11` for Q, and `10` for H), followed by the 3 bits of the mask pattern. They are encoded by the $(15, 5)$ [BCH code](/ref/BCHCode) with generator polynomial
$$
    g(X) = X^{10} + X^8 + X^5 + X^4 + X^2 + X + 1,
$$
and the resulting codeword is added to the fixed mask $101010000010010$, so that the format information is never all zeros. For more details, see <cite>ISO24, Sec. 7.9</cite>.

In Komm, this is the BCH code with $\mu = 4$ and $\delta = 7$:

```pycon
>>> import numpy as np
>>> import komm

>>> code = komm.BCHCode(mu=4, delta=7)
>>> (code.length, code.dimension, code.minimum_distance())
(15, 5, 7)
>>> code.generator_polynomial
BinaryPolynomial(0b10100110111)

```

Komm stores the coefficient of $X^i$ at position $i$ of a word, and its encoder puts the 10 parity bits on the left. The specification writes the bits from $X^{14}$ down to $X^0$, so that the 5 information bits come first. The functions below reverse the bits and add the mask; to decode, they use the [Berlekamp decoder](/ref/BerlekampDecoder):

```pycon
>>> def bits(word):
...     return "".join(map(str, word))

>>> mask = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
>>> decoder = komm.BerlekampDecoder(code)

>>> def encode(message):
...     return code.encode(message[::-1])[::-1] ^ mask

>>> def decode(word):
...     return decoder.decode((word ^ mask)[::-1])[::-1]

```

For example, level M with mask pattern 5:

```pycon
>>> message = [0, 0, 1, 0, 1]
>>> word = encode(message)
>>> print(bits(word))
100000011001110

```

A symbol carries two copies of the format information: one around the top-left finder pattern, and another split between the other two. The figure shows the word above in a symbol of version 1, with the data area in gray. The module labeled $i$ holds the coefficient of $X^i$.

<figure markdown>
  ![The two copies of the format information in a QR code symbol of version 1.](/fig/qr_format.svg)
</figure>

All the 32 format words, with one row per mask pattern and one column per level (L, M, Q, and H):

```pycon
>>> levels = [[0, 1], [0, 0], [1, 1], [1, 0]]  # L, M, Q, H
>>> for i in range(8):
...     pattern = komm.to_binary(i, width=3, bit_order="MSB-first")
...     words = [encode(level + pattern) for level in levels]
...     print(bits(pattern), *[bits(word) for word in words])
000 111011111000100 101010000010010 011010101011111 001011010001001
001 111001011110011 101000100100101 011000001101000 001001110111110
010 111110110101010 101111001111100 011111100110001 001110011100111
011 111100010011101 101101101001011 011101000000110 001100111010000
100 110011000101111 100010111111001 010010010110100 000011101100010
101 110001100011000 100000011001110 010000110000011 000001001010101
110 110110001000001 100111110010111 010111011011010 000110100001100
111 110100101110110 100101010100000 010101111101101 000100000111011

```

The mask changes no distance, so the format words differ in at least 7 positions, and the decoder corrects up to 3 errors:

```pycon
>>> received = word.copy()
>>> received[[1, 7, 12]] ^= 1  # Three errors
>>> print(bits(received))
110000001001010
>>> decode(received)
array([0, 0, 1, 0, 1])

```

The same holds for all the $1 + 15 + 105 + 455 = 576$ error patterns of weight at most 3:

```pycon
>>> from itertools import combinations

>>> count = 0
>>> for w in range(4):
...     for positions in combinations(range(15), w):
...         received = word.copy()
...         received[list(positions)] ^= 1
...         count += np.array_equal(decode(received), message)
>>> count
576

```
