# Lempel–Ziv 78 and the entropy rate

The [Lempel–Ziv 78 (LZ78) code](/ref/LempelZiv78Code) compresses a sequence without any knowledge of the source. It parses the sequence into _phrases_, each one a previous phrase followed by one more symbol, and represents each phrase by a token $(p, x)$, where $p$ is the index of the previous phrase (with $0$ for the empty phrase) and $x$ is the new symbol. For a stationary ergodic source, its rate approaches the _entropy rate_ of the source as the length of the sequence grows <cite>CT06, Sec. 13.5.2</cite>. The [extended Huffman codes](extended-huffman.md), on the other hand, need the statistics of the source.

Consider the [Markov source](/ref/MarkovSource) from <cite>Gal08, Sec. 2.8</cite>, whose state diagram is depicted in the figure below. Each transition is labeled with the emitted symbol and its probability.

<figure markdown>
  ![Markov source with four states.](/fig/lz78_source.svg)
</figure>

The state is given by the last two emitted bits, with the states $0$, $1$, $2$, and $3$ standing for $00$, $01$, $10$, and $11$, respectively. After two equal bits, the source repeats the last bit with probability $0.9$; otherwise, it emits each bit with probability $1/2$. Its entropy rate is about $0.557$ bits per symbol. Each bit, however, is equally likely to be $0$ or $1$, so that a code for single symbols cannot compress the source:

```pycon
>>> import numpy as np
>>> import komm

>>> komm.global_rng.set(np.random.default_rng(seed=42))

>>> source = komm.MarkovSource(
...     transitions=[[0, 1], [2, 3], [0, 1], [2, 3]],
...     pmfs=[[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]],
... )
>>> source.entropy_rate()  # doctest: +FLOAT_CMP
0.557496327991068
>>> pmf = source.initial_pmf @ source.pmfs  # Of each symbol
>>> pmf
array([0.5, 0.5])
>>> komm.entropy(pmf)  # doctest: +FLOAT_CMP
1.0

```

The source emits long runs of zeros and ones, so the phrases grow quickly. The first $24$ symbols below make the $8$ phrases $1$, $11$, $0$, $111$, $1111$, $11111$, $01$, and $111111$:

```pycon
>>> x = source.emit(100_000)
>>> x[:24]
array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
>>> lz78 = komm.LempelZiv78Code(2)
>>> lz78.source_to_tokens(x[:24])
[(0, 1), (1, 1), (0, 0), (2, 1), (4, 1), (5, 1), (3, 1), (6, 1)]

```

The rate of the code, in bits per source symbol, is the length of the output divided by the length of the input. On longer and longer prefixes of the sequence, it decreases toward the entropy rate, but slowly. For contrast, a fair coin, a memoryless source with entropy rate $1$, gets expanded: there is nothing to learn, and the pointers cost bits.

```pycon
>>> coin = komm.DiscreteMemorylessSource(2)
>>> z = coin.emit(100_000)

>>> ns = np.logspace(2, 5, 7).astype(int)
>>> rates_source = [lz78.encode(x[:n]).size / n for n in ns]
>>> rates_coin = [lz78.encode(z[:n]).size / n for n in ns]

```

To plot the curves, with [Matplotlib](https://matplotlib.org), and the entropy rates as dashed lines:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.semilogx(ns, rates_source, "o-", label="Markov source")
ax.semilogx(ns, rates_coin, "s-", label="Fair coin")
ax.axhline(source.entropy_rate(), color="C0", ls="--")
ax.axhline(1, color="C1", ls="--")
ax.set(xlabel="Sequence length", ylabel="Rate (bits per symbol)", ylim=(0, 1.5))
ax.grid(which="both")
ax.legend()
plt.show()
```

<figure markdown>
  ![Rate of the Lempel–Ziv 78 code for a Markov source and a fair coin.](/fig/lz78_entropy_rate.svg)
</figure>

The phrases form a tree, in which each phrase hangs from the previous phrase it extends. The figure below shows the trees for the first $300$ symbols of each sequence, with the edges of $0$ in blue and the edges of $1$ in red. For the coin, the tree is full and shallow, and takes $67$ tokens; for the Markov source, it is sparse and deep, with long branches from the runs, and takes only $50$ tokens.

<figure markdown>
  ![Phrase trees of the Lempel–Ziv 78 code for a Markov source and a fair coin.](/fig/lz78_trees.svg)
</figure>

The universality has a price. With $10^5$ symbols, the rate of the LZ78 code is $0.711$ bits per symbol, close to the $0.7125$ bits per symbol of a Huffman code for blocks of $4$ symbols designed for this source.
