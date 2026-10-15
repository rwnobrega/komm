# Extended Huffman codes

The $k$-th _extension_ of a [discrete memoryless source](/ref/DiscreteMemorylessSource) with pmf $p$ over $\mathcal{X}$ is the source that emits blocks of $k$ symbols of the original source, that is, the source with alphabet $\mathcal{X}^k$ and pmf $p_k(x_0, \ldots, x_{k-1}) = p(x_0) \cdots p(x_{k-1})$. Its [Huffman code](/ref/HuffmanCode) is an _extended Huffman code_ of the original source. If $\bar{n}_k$ is the average codeword length of this code, then its rate $R_k = \bar{n}_k / k$, in bits per source symbol, satisfies
$$
    \mathrm{H} \leq R_k < \mathrm{H} + \frac{1}{k},
$$
where $\mathrm{H}$ is the [entropy](/ref/entropy) of the source. So the rate approaches the entropy as $k$ grows. For more details, see <cite>Say06, Sec. 3.2.4</cite>.

Consider the source with pmf $p = (0.8, 0.02, 0.18)$ over $\mathcal{X} = \\{ 0, 1, 2 \\}$. Its Huffman code has rate $1.2$ bits per symbol, far from the entropy:

```pycon
>>> import numpy as np
>>> import komm

>>> pmf = [0.8, 0.02, 0.18]
>>> komm.entropy(pmf)  # doctest: +FLOAT_CMP
0.8157272136052185

>>> code = komm.HuffmanCode(pmf)
>>> code.codewords
[(0,), (1, 1), (1, 0)]
>>> code.average_length(pmf)  # doctest: +FLOAT_CMP
1.2

```

The pmf of the second extension is the Kronecker product of $p$ with itself, in which the pair $(x_0, x_1)$ has index $3 x_0 + x_1$. Its Huffman code has average codeword length $1.7228$ bits per pair, that is, rate $0.8614$ bits per symbol:

```pycon
>>> pmf2 = np.kron(pmf, pmf)
>>> code2 = komm.HuffmanCode(pmf2)
>>> for i, word in enumerate(code2.codewords):
...     print(divmod(i, 3), f"{pmf2[i]:.4f}", "".join(map(str, word)))
(0, 0) 0.6400 0
(0, 1) 0.0160 10101
(0, 2) 0.1440 11
(1, 0) 0.0160 101000
(1, 1) 0.0004 10100101
(1, 2) 0.0036 1010011
(2, 0) 0.1440 100
(2, 1) 0.0036 10100100
(2, 2) 0.0324 1011
>>> code2.average_length(pmf2)  # doctest: +FLOAT_CMP
1.7228
>>> code2.average_length(pmf2) / 2  # doctest: +FLOAT_CMP
0.8614

```

The pmf of the $k$-th extension is the Kronecker product of $k$ copies of $p$. The rate approaches the entropy, though not monotonically:

```pycon
>>> from functools import reduce

>>> for k in range(1, 7):
...     pmf_k = reduce(np.kron, [pmf] * k)
...     code_k = komm.HuffmanCode(pmf_k)
...     print(k, f"{code_k.average_length(pmf_k) / k:.4f}")
1 1.2000
2 0.8614
3 0.8209
4 0.8302
5 0.8281
6 0.8194

```

To encode a sequence of source symbols with the code of the second extension, group the symbols in pairs, and replace each pair $(x_0, x_1)$ by its index $u = 3 x_0 + x_1$:

```pycon
>>> x = np.array([0, 0, 0, 2, 0, 0, 2, 0])
>>> pairs = x.reshape(-1, 2)
>>> pairs
array([[0, 0],
       [0, 2],
       [0, 0],
       [2, 0]])
>>> u = 3 * pairs[:, 0] + pairs[:, 1]
>>> u
array([0, 2, 0, 6])
>>> y = code2.encode(u)
>>> y
array([0, 1, 1, 0, 1, 0, 0])

```

To decode, do the reverse: $x_0$ and $x_1$ are the quotient and the remainder of the division of $u$ by $3$:

```pycon
>>> u_hat = code2.decode(y)
>>> u_hat
array([0, 2, 0, 6])
>>> pairs_hat = np.column_stack([u_hat // 3, u_hat % 3])
>>> x_hat = pairs_hat.ravel()
>>> x_hat
array([0, 0, 0, 2, 0, 0, 2, 0])

```

In general, the index of a block $(x_0, \ldots, x_{k-1})$ is the number whose base-$3$ digits are $x_0, \ldots, x_{k-1}$, that is, $u = 3^{k-1} x_0 + \cdots + 3 x_{k-2} + x_{k-1}$. For any $k$, the NumPy functions [`ravel_multi_index`](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html) and [`unravel_index`](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html) do the two conversions, with shape `(3,) * k`. For pairs, they give the same as above:

```pycon
>>> np.ravel_multi_index(pairs.T, (3, 3))
array([0, 2, 0, 6])
>>> np.column_stack(np.unravel_index(u_hat, (3, 3))).ravel()
array([0, 0, 0, 2, 0, 0, 2, 0])

```
