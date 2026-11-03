# Text compression

This page compresses the text of _Alice's Adventures in Wonderland_, by Lewis Carroll, with a [Huffman code](/ref/HuffmanCode) and four Lempel–Ziv codes: [LZ77](/ref/LempelZiv77Code), [LZSS](/ref/LempelZivSSCode), [LZ78](/ref/LempelZiv78Code), and [LZW](/ref/LempelZivWelchCode). The text is the file [`alice29.txt`](/gallery/alice29.txt) of the [Canterbury corpus](https://corpus.canterbury.ac.nz/), a benchmark for lossless compression; to run the code below, save it in the working directory. Read as bytes, the text is a sequence of symbols from an alphabet of $256$ symbols:

```pycon
>>> import numpy as np
>>> import komm

>>> x = np.fromfile("alice29.txt", dtype=np.uint8)
>>> x.size
152089
>>> x[253:315].tobytes()
b'Alice was beginning to get very tired of sitting by her sister'

```

Only $74$ of the $256$ symbols appear in the text, and their frequencies give the pmf of the source. The $15$ most frequent characters are the space, the line breaks (each line ends with CR and LF), and the twelve letters of [_etaoin shrdlu_](https://en.wikipedia.org/wiki/Etaoin_shrdlu), the classic ranking of the most frequent letters in English, in a slightly different order:

```pycon
>>> np.unique(x).size
74
>>> pmf = np.bincount(x, minlength=256) / x.size
>>> ranking = np.argsort(pmf, kind="stable")[::-1]  # Most frequent first
>>> bytes(ranking[:15].tolist())
b' etaohnisrdl\r\nu'

```

To plot the pmf, with [Matplotlib](https://matplotlib.org), and some groups of characters in color:

```python
import matplotlib.pyplot as plt

groups = {
    "Lowercase": range(ord("a"), ord("z") + 1),
    "Uppercase": range(ord("A"), ord("Z") + 1),
    "Space": [ord(" ")],
    "Line breaks": [ord("\r"), ord("\n")],
}

fig, ax = plt.subplots()
ax.bar(range(128), pmf[:128], width=1, color="gray")
for label, codes in groups.items():
    ax.bar(codes, pmf[codes], width=1, label=label)
ax.set(xlabel="Byte value", ylabel="Probability")
ax.legend()
plt.show()
```

<figure markdown>
  ![Pmf of the characters of the text, with groups of characters in color.](/fig/text_compression_pmf.svg)
</figure>

The lowercase letters take $68\%$ of the mass, and the space $19\%$. The uppercase letters take $3\%$, the line breaks $5\%$, and the other characters, in gray, mostly punctuation marks, the remaining $5\%$.

The Huffman code designed for this pmf has rate $4.61$ bits per character, close to the [entropy](/ref/entropy) of $4.57$ bits per character:

```pycon
>>> komm.entropy(pmf)  # doctest: +FLOAT_CMP
4.567680212177265
>>> huffman = komm.HuffmanCode(pmf)
>>> huffman.average_length(pmf)  # doctest: +FLOAT_CMP
4.612450604580214

```

The decoder also needs the code, which depends on the text. A canonical Huffman code is given by its codeword lengths, which take at most $256$ bytes here, about $0.01$ bits per character.

The Lempel–Ziv codes, on the other hand, need no knowledge of the text: they replace repeated strings, such as words and phrases, by references to earlier occurrences. For more details, see <cite>Say06, Sec. 5.4</cite>. Below, the sliding window of LZ77 and LZSS has a search buffer of size $S = 2^{16}$ and a lookahead buffer of size $L = 16$. The rate of each code is the length of its output divided by the length of the text:

```pycon
>>> window = dict(search_size=2**16, lookahead_size=16)
>>> codes = {
...     "Huffman": huffman,
...     "LZ77": komm.LempelZiv77Code(**window, source_cardinality=256),
...     "LZSS": komm.LempelZivSSCode(**window, source_cardinality=256),
...     "LZ78": komm.LempelZiv78Code(source_cardinality=256),
...     "LZW": komm.LempelZivWelchCode(source_cardinality=256),
... }
>>> outputs = {name: code.encode(x) for name, code in codes.items()}
>>> rates = {name: y.size / x.size for name, y in outputs.items()}

```

All the codes are lossless: decoding each output gives back the text:

```pycon
>>> x_hats = [codes[name].decode(y) for name, y in outputs.items()]
>>> all(np.array_equal(x_hat, x) for x_hat in x_hats)
True

```

To plot the rates, with the entropy as a dashed line:

```python
fig, ax = plt.subplots()
bars = ax.barh(list(rates.keys()), list(rates.values()))
ax.bar_label(bars, fmt="%.2f", label_type="center", color="white")
ax.axvline(komm.entropy(pmf), color="k", ls="--", label="Entropy")
ax.set(xlabel="Rate (bits per character)", xlim=(0, 5))
ax.invert_yaxis()
ax.legend(loc="lower right")
plt.show()
```

<figure markdown>
  ![Rates of the Huffman and Lempel–Ziv codes for the text, with the entropy.](/fig/text_compression.svg)
</figure>

The four Lempel–Ziv codes go below the entropy of the characters, with no contradiction: the entropy bounds the rate of codes that encode each character on its own, like the Huffman code. In a text, each character depends on the previous ones, and the Lempel–Ziv codes exploit this memory, as in [Lempel–Ziv 78 and the entropy rate](lz78-entropy-rate.md). LZSS and LZW improve on LZ77 and LZ78, whose tokens always carry a source symbol.

The size $S$ of the search buffer sets a trade-off: a larger buffer finds longer matches, but each pointer costs more bits. For this text, $S = 2^{16}$ gives the lowest rates among the powers of two. With $S = 2^{10}$, the rate of LZ77 is $5.12$ bits per character, worse than that of the Huffman code.

For comparison, Deflate, the algorithm of gzip and PNG, compresses this file to about $2.85$ bits per character. It is, in essence, an LZSS code with [Huffman codes for the tokens](lzss-huffman.md).
