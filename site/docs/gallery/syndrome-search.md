# Syndrome decoding without a table

Given a received word $r$, a decoder can try error patterns $e$ from the most likely to the least likely, and stop at the first one for which $r + e$ is a codeword. Since the noise does not depend on the transmitted codeword, the result is the maximum-likelihood codeword. Over the [binary symmetric channel](/ref/BinarySymmetricChannel) with crossover probability $p < 1/2$, an error pattern of Hamming weight $w$ has probability $p^w (1 - p)^{n - w}$, which decreases with $w$, so the decoder tries the error patterns in order of increasing weight <cite>Bar97, Secs. 3.3.1 and 3.4.2</cite>. This search is now known as _guessing random additive noise decoding_ (GRAND) <cite>DLM19</cite>.

The function below generates the error patterns of length $n$ in order of increasing weight and, within each weight, with the positions of the ones in lexicographic order:

```pycon
>>> from itertools import combinations, islice, product

>>> import numpy as np
>>> import komm

>>> def error_patterns(n):
...     for w in range(n + 1):
...         for ones in combinations(range(n), w):
...             e = np.zeros(n, dtype=int)
...             e[list(ones)] = 1
...             yield e

>>> np.array(list(islice(error_patterns(5), 8)))
array([[0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [1, 1, 0, 0, 0],
       [1, 0, 1, 0, 0]])

```

With [check matrix](/ref/BlockCode#check_matrix) $H$, the word $r + e$ is a codeword if and only if its syndrome $(r + e) H^\top$ is zero. The decoder below takes a received word, and returns the decoded codeword and the number of error patterns tried:

```pycon
>>> def decode(code, r):
...     H = code.check_matrix
...     for tries, e in enumerate(error_patterns(code.length), start=1):
...         s = (r ^ e) @ H.T % 2
...         if not s.any():
...             return r ^ e, tries

>>> code = komm.BlockCode(generator_matrix=[
...     [1, 0, 0, 1, 1],
...     [0, 1, 1, 1, 0],
... ])
>>> decode(code, [1, 1, 1, 1, 1])
(array([1, 1, 1, 0, 1]), 5)
>>> decode(code, [1, 1, 0, 0, 0])
(array([0, 0, 0, 0, 0]), 7)

```

For the first word, the fifth error pattern, $e = 00010$, gives a zero syndrome; for the second, the seventh, $e = 11000$. For a [linear block code](/ref/BlockCode), the error pattern found is the leader of the coset of $r$ <cite>LC04, Sec. 3.5</cite>, which the [syndrome table decoder](/ref/SyndromeTableDecoder) looks up in a table with $2^m$ entries, one per syndrome (see [Standard array and syndrome decoding](standard-array.md)). In each coset, the search stops at the first minimal-weight word in this order, which is the coset leader that [`coset_leaders`](/ref/BlockCode#coset_leaders) returns. So the decoder above agrees with the syndrome table decoder on all the $2^5$ possible received words:

```pycon
>>> decoder = komm.SyndromeTableDecoder(code)
>>> words = np.array(list(product([0, 1], repeat=5)))
>>> decoded = np.array([decode(code, r)[0] for r in words])
>>> np.array_equal(decoded, decoder.decode_to_codeword(words))
True

```

The search needs no table, but the number of error patterns it tries is random, and grows with the noise. The simulation below counts the tries over the binary symmetric channel, with the [Golay code](/ref/GolayCode), for $1000$ codewords at each value of $p$:

```pycon
>>> rng = np.random.default_rng(seed=42)
>>> komm.global_rng.set(rng)

>>> code = komm.GolayCode()
>>> source = komm.DiscreteMemorylessSource(2)
>>> ps = np.logspace(-3, -1, 9)
>>> mean_tries, max_tries = [], []
>>> for p in ps:
...     channel = komm.BinarySymmetricChannel(p)
...     v = code.encode(source.emit((1000, code.dimension)))
...     tries = [decode(code, r)[1] for r in channel.transmit(v)]
...     mean_tries.append(np.mean(tries))
...     max_tries.append(np.max(tries))

```

Since the Golay code is perfect, every word of weight at most $3$ is a coset leader, so the search never tries more than $2^{11} = 2048$ error patterns. To plot the mean and the maximum number of tries, with [Matplotlib](https://matplotlib.org), and this bound as a dashed line:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.loglog(ps, mean_tries, "o-", label="Mean")
ax.loglog(ps, max_tries, "s-", label="Maximum")
ax.axhline(2048, color="k", ls="--")
ax.set(xlabel="$p$", ylabel="Error patterns tried", ylim=(1, 1e4))
ax.grid(which="both")
ax.legend()
plt.show()
```

<figure markdown>
  ![Mean and maximum number of error patterns tried by the search for the Golay code over the binary symmetric channel.](/fig/syndrome_search.svg)
</figure>

The mean goes from about $1$ for $p = 0.001$ to about $500$ for $p = 0.1$, and the maximum gets close to the bound from about $p = 0.03$ on. For comparison, the [exhaustive codeword decoder](/ref/ExhaustiveCodewordDecoder) compares each received word with all the $2^{12} = 4096$ codewords, whatever the noise.
