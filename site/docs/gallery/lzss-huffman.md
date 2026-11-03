# Huffman codes for LZSS tokens

[Deflate](https://en.wikipedia.org/wiki/Deflate), the algorithm of gzip and PNG, is, in essence, an [LZSS code](/ref/LempelZivSSCode) whose tokens go through [Huffman codes](/ref/HuffmanCode). This page applies the two main ideas of Deflate to the LZSS code of [Text compression](text-compression.md), without the details of the format, which are in <cite>Deu96</cite>.

As there, the source is the text of _Alice's Adventures in Wonderland_, in the file [`alice29.txt`](/gallery/alice29.txt) (to run the code below, save it in the working directory), and the sliding window has a search buffer of size $S = 2^{16}$ and a lookahead buffer of size $L = 16$. The text starts with four line breaks (CR and LF) and $16$ spaces, which take five tokens:

```pycon
>>> import numpy as np
>>> import komm

>>> x = np.fromfile("alice29.txt", dtype=np.uint8)
>>> lzss = komm.LempelZivSSCode(
...     search_size=2**16,
...     lookahead_size=16,
...     source_cardinality=256,
... )
>>> tokens = lzss.source_to_tokens(x)
>>> tokens[:5]
[(0, 13), (0, 10), (1, 2, 6), (0, 32), (1, 1, 15)]

```

Each token is a literal $(0, x)$, where $x$ is a byte, or a reference $(1, p, \ell)$, where $p$ is the pointer and $\ell$ is the length of the match; in this text, most tokens are references. The LZSS code writes them with fixed widths: a literal takes $1 + 8 = 9$ bits, for the flag and the byte, and a reference takes $1 + 4 = 5$ bits, for the flag and the length, plus $16$ bits for the pointer. The rate is $3.21$ bits per character:

```pycon
>>> references = sum(t[0] for t in tokens)
>>> literals = len(tokens) - references
>>> literals, references
(3083, 21934)
>>> fixed = {
...     "Literals": 9 * literals,
...     "Lengths": 5 * references,
...     "Pointers": 16 * references,
... }
>>> sum(fixed.values()) / x.size  # doctest: +FLOAT_CMP
3.2110211783889695

```

The first idea of Deflate is a single Huffman code for the bytes and the lengths. The symbol of a literal is its byte $x$, and that of a reference is $256 + \ell - 1$, so that the symbol itself tells whether a pointer follows, and the flag goes away. The alphabet has $256 + 16 = 272$ symbols: $[0 : 256)$ for the bytes and $[256 : 272)$ for the lengths:

```pycon
>>> symbols = [256 + t[2] - 1 if t[0] else t[1] for t in tokens]
>>> symbols[:5]
[13, 10, 261, 32, 270]

```

The second idea is a Huffman code for the sizes of the pointers. A pointer goes as its number of bits, in a second Huffman code, followed by its bits after the leading one, in plain binary. For example, the pointer $52$, which is `110100` in binary, goes as the size $6$ followed by `10100`. The [Elias gamma code](/ref/EliasGammaCode) does the same, with a unary code for the number of bits. The sizes are the symbols of the second Huffman code:

```pycon
>>> pointers = [t[1] for t in tokens if t[0]]
>>> sizes = [p.bit_length() for p in pointers]
>>> pointers[:5], sizes[:5]
([2, 1, 52, 1, 43], [2, 1, 6, 1, 6])

```

Both Huffman codes are designed for the text, with only the symbols that occur. The decoder also needs them. As canonical codes, they are given by their codeword lengths, which go in a header: one byte for each of the $272$ symbols and the $17$ sizes, from $1$ bit (pointer $1$) to $17$ bits (pointer $2^{16}$). With the header, the rate is:

```pycon
>>> def huffman_lengths(symbols):
...     values, counts = np.unique(symbols, return_counts=True)
...     code = komm.HuffmanCode(counts / counts.sum())
...     return dict(zip(values.tolist(), code.lengths))

>>> symbol_lengths = huffman_lengths(symbols)
>>> size_lengths = huffman_lengths(sizes)
>>> huffman = {
...     "Literals": sum(symbol_lengths[s] for s in symbols if s < 256),
...     "Lengths": sum(symbol_lengths[s] for s in symbols if s >= 256),
...     "Pointers": sum(size_lengths[k] + k - 1 for k in sizes),
...     "Header": 8 * (272 + 17),
... }
>>> sum(huffman.values()) / x.size  # doctest: +FLOAT_CMP
2.8866979202966685

```

The rate drops to $2.89$ bits per character, of which the header takes only $0.015$. To compare the parts of the rate, with [Matplotlib](https://matplotlib.org):

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 2.4), layout="constrained")
methods = ["Fixed widths", "Huffman codes"]
left = np.zeros(2)
for part in huffman:
    rates = np.array([fixed.get(part, 0), huffman[part]]) / x.size
    bars = ax.barh(methods, rates, left=left, label=part)
    if part != "Header":  # Too thin for a label
        ax.bar_label(bars, fmt="%.2f", label_type="center", color="white")
    left += rates
ax.bar_label(bars, labels=[f"{r:.2f}" for r in left], padding=3)
ax.set(xlabel="Rate (bits per character)", xlim=(0, 3.5))
ax.invert_yaxis()
fig.legend(loc="outside upper center", ncols=4)
plt.show()
```

<figure markdown>
  ![Parts of the rate of the LZSS code, with fixed widths and with Huffman codes.](/fig/lzss_huffman.svg)
</figure>

With fixed widths and with Huffman codes, the bits per literal, length, and pointer are:

```pycon
>>> for part in fixed:
...     n = literals if part == "Literals" else references
...     print(f"{part:8} {fixed[part] / n:5.2f} {huffman[part] / n:5.2f}")
Literals  9.00  8.88
Lengths   5.00  3.59
Pointers 16.00 15.07

```

The flag goes into the symbols. The lengths, common in this text, get short codewords, and drop from $5$ to $3.59$ bits each. The literals, rare in this text, get long codewords, and keep about their $9$ bits. The pointers drop only from $16$ to $15.07$ bits: most matches lie thousands of characters back, so the pointers are large, and the Huffman code compresses only their sizes. The bits after the leading one go in plain binary, and take $1.70$ of the $2.89$ bits per character:

```pycon
>>> sum(k - 1 for k in sizes) / x.size  # doctest: +FLOAT_CMP
1.6975652414047038

```

Deflate adds a few refinements to these ideas, and zlib, its most common implementation, compresses this file to about $2.85$ bits per character.
