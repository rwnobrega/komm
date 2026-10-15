# Standard array and syndrome decoding

The _standard array_ (or _Slepian array_) of a [linear block code](/ref/BlockCode) is a table with $2^m$ rows and $2^k$ columns, where $m$ is the redundancy and $k$ is the dimension of the code. Each row is a coset of the code: the first row holds [all the codewords](/ref/BlockCode#codewords) of the code, and the first column holds the [coset leaders](/ref/BlockCode#coset_leaders) (words of minimal weight in each coset). The entry in row $i$ and column $j$ is the sum of the $i$-th coset leader and the $j$-th codeword. For more details, see <cite>LC04, Sec. 3.5</cite>.

```pycon
>>> import numpy as np
>>> import komm

>>> def bits(word):
...     return "".join(map(str, word))

>>> code = komm.BlockCode(generator_matrix=[
...     [1, 0, 0, 1, 1],
...     [0, 1, 1, 1, 0],
... ])
>>> array = code.coset_leaders()[:, np.newaxis] ^ code.codewords()
>>> array.shape
(8, 4, 5)

>>> for row in array:
...     print(*[bits(word) for word in row])
00000 10011 01110 11101
00100 10111 01010 11001
00010 10001 01100 11111
01000 11011 00110 10101
00001 10010 01111 11100
11000 01011 10110 00101
10000 00011 11110 01101
10100 00111 11010 01001

```

Row $i$ corresponds to the syndrome obtained by expressing $i$ in binary (LSB-first), and column $j$ to the message obtained by expressing $j$ in binary (LSB-first):

```pycon
>>> for i, leader in enumerate(array[:, 0]):
...     print(i, bits(code.check(leader)), bits(leader))
0 000 00000
1 100 00100
2 010 00010
3 110 01000
4 001 00001
5 101 11000
6 011 10000
7 111 10100

>>> for j, codeword in enumerate(array[0, :]):
...     print(j, bits(code.inverse_encode(codeword)), bits(codeword))
0 00 00000
1 10 10011
2 01 01110
3 11 11101

```

To decode a received word, find its row from the syndrome, and add the coset leader of that row. The result is the codeword at the top of its column. The [syndrome table decoder](/ref/SyndromeTableDecoder) does the same, without the array:

```pycon
>>> r = [1, 1, 1, 1, 1]
>>> i = komm.from_binary(code.check(r))
>>> r ^ array[i, 0]
array([1, 1, 1, 0, 1])

>>> decoder = komm.SyndromeTableDecoder(code)
>>> decoder.decode_to_codeword(r)
array([1, 1, 1, 0, 1])

```
