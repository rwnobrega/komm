# Nadler code

Nadler found by hand a binary code with $M = 32$ codewords of length $n = 12$ and minimum distance $5$ <cite>Nad62</cite>. The code is not linear, and Nadler was not sure that a linear code with these parameters could be found. In fact, none exists: the largest minimum distance of a linear code with these parameters is $4$ (see [codetables.de](https://www.codetables.de/BKLC/BKLC.php?q=2&n=12&k=5)).

He listed the codewords in $16$ pairs, where the two words of each pair have the same first $5$ bits and complementary last $7$ bits. The first $5$ bits run over the $16$ words of length $5$ and even weight. The last $7$ bits run over a cyclic Hamming code in the first $8$ pairs, in the order of its cyclic shifts, and over a translate of another Hamming code in the last $8$:

```pycon
>>> from itertools import product

>>> import numpy as np
>>> import komm

>>> table = [
...     "00000 1100010", "00000 0011101", "10001 1000101", "10001 0111010",
...     "01001 0001011", "01001 1110100", "11011 0010110", "11011 1101001",
...     "00011 0101100", "00011 1010011", "11000 1011000", "11000 0100111",
...     "01010 0110001", "01010 1001110", "10010 1111111", "10010 0000000",
...     "00101 0000110", "00101 1111001", "10100 0010011", "10100 1101100",
...     "01100 1000001", "01100 0111110", "11110 0001101", "11110 1110010",
...     "00110 1010100", "00110 0101011", "11101 0100000", "11101 1011111",
...     "01111 0011000", "01111 1100111", "10111 1001010", "10111 0110101",
... ]
>>> nadler = np.array([[int(b) for b in row if b != " "] for row in table])

```

If the code were linear, or a linear code plus a fixed word, the sum of any three codewords would be a codeword. The sum of the first words of the first three pairs is not:

```pycon
>>> x = nadler[0] ^ nadler[2] ^ nadler[4]
>>> any(np.array_equal(x, v) for v in nadler)
False

```

Below, entry $i$ of `codeword_distance_distribution(nadler)` is the number of codewords at distance $i$ from a codeword, averaged over the codewords: the _distance distribution_ of the code. Its first nonzero entry after entry $0$ gives the minimum distance, $5$:

```pycon
>>> def codeword_distance_distribution(codewords):
...     M, n = codewords.shape
...     distances = np.sum(codewords[:, np.newaxis] ^ codewords, axis=-1)
...     return np.bincount(distances.ravel(), minlength=n + 1) / M

>>> codeword_distance_distribution(nadler)
array([ 1.,  0.,  0.,  0.,  0., 11., 13.,  2.,  1.,  3.,  1.,  0.,  0.])

```

For a nonlinear code, the entries need not be integers. Here they are, because Nadler's code is _distance invariant_: every codeword has the same number of codewords at each distance, as in any linear code.

For comparison, puncturing the first-order [Reed–Muller code](/ref/ReedMullerCode) of length $16$ at positions $1$, $2$, $4$, and $8$ (counting from $0$) gives a linear code with the same length and number of codewords, and minimum distance $4$, the largest possible. For a linear code, the distances from a codeword $v$ are the weights of $v + v'$, which runs over the whole code as $v'$ does, so the distance distribution is the [codeword weight distribution](/ref/BlockCode#codeword_weight_distribution):

```pycon
>>> rm = komm.ReedMullerCode(mu=4, rho=1)
>>> G = np.delete(rm.generator_matrix, [1, 2, 4, 8], axis=1)
>>> linear = komm.BlockCode(generator_matrix=G)
>>> linear.codeword_weight_distribution()
[1, 0, 0, 0, 1, 8, 12, 8, 1, 0, 0, 0, 1]

```

Over the [binary symmetric channel](/ref/BinarySymmetricChannel) with crossover probability $p < 1/2$, the maximum-likelihood decoder picks a codeword nearest to the received word $r$. With equiprobable messages, its block error probability is
$$
    P_\mathrm{B} = 1 - \frac{1}{M} \sum_{r \in \mathbb{B}^n} p^{d(r)} (1 - p)^{n - d(r)} = 1 - \sum_{i=0}^n \alpha_i \, p^i (1 - p)^{n - i},
$$
where $d(r)$ is the distance from $r$ to the code and $\alpha_i$ is the number of words $r$ with $d(r) = i$ per codeword. Counting the $2^{12} = 4096$ possible received words gives $\alpha_0, \alpha_1, \ldots, \alpha_n$ for Nadler's code:

```pycon
>>> def received_distance_distribution(codewords):
...     M, n = codewords.shape
...     received = np.array(list(product([0, 1], repeat=n)))
...     distances = np.sum(received[:, np.newaxis] ^ codewords, axis=-1)
...     d = np.min(distances, axis=-1)
...     return np.bincount(d, minlength=n + 1) / M

>>> alpha_nadler = received_distance_distribution(nadler)
>>> alpha_nadler
array([ 1., 12., 66., 46.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

```

For a nonlinear code, the $\alpha_i$ need not be integers. For a linear code, they are the [coset leader weight distribution](/ref/BlockCode#coset_leader_weight_distribution):

```pycon
>>> alpha_linear = linear.coset_leader_weight_distribution()
>>> alpha_linear
[1, 12, 63, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0]

```

Both codes correct all single errors, but only Nadler's code corrects all the $\binom{12}{2} = 66$ double errors. To plot the block error probabilities of the two codes, with [Matplotlib](https://matplotlib.org):

```python
import matplotlib.pyplot as plt


def block_error(alpha, p):
    n = len(alpha) - 1
    i = np.arange(n + 1)
    return 1 - np.sum(alpha * p**i * (1 - p) ** (n - i))


ps = np.logspace(-3, np.log10(0.5))
fig, ax = plt.subplots()
ax.loglog(ps, [block_error(alpha_nadler, p) for p in ps], label="Nadler's code")
ax.loglog(ps, [block_error(alpha_linear, p) for p in ps], label="Linear code")
ax.set(xlabel="$p$", ylabel=r"$P_\mathrm{B}$")
ax.grid(which="both")
ax.legend()
plt.show()
```

<figure markdown>
  ![Block error probabilities of Nadler's code and of the linear code over the BSC.](/fig/nadler_code.svg)
</figure>

The $\alpha_i$ of the two codes differ by $3$, $-6$, and $3$ for $i = 2$, $3$, and $4$. So the difference between the block error probabilities is
$$
    3p^2 (1 - p)^{10} - 6p^3 (1 - p)^9 + 3p^4 (1 - p)^8 = 3p^2 (1 - p)^8 (1 - 2p)^2,
$$
which is positive for all $p < 1/2$.

Van Lint later published "a new description of the Nadler code" <cite>vL72</cite>. Up to equivalence, his code comes from Nadler's by appending a parity bit and puncturing at position $2$. It has the same parameters, but other distances between codewords, with none at distances $7$ and $10$. So it is not equivalent to Nadler's code:

```pycon
>>> parity = np.sum(nadler, axis=1, keepdims=True) % 2
>>> vanlint = np.delete(np.hstack([nadler, parity]), 2, axis=1)
>>> codeword_distance_distribution(vanlint)
array([ 1.,  0.,  0.,  0.,  0., 12., 12.,  0.,  3.,  4.,  0.,  0.,  0.])

```

Still, it has the same $\alpha_i$ as Nadler's code, so over the BSC the two codes have the same block error probability for every $p$:

```pycon
>>> received_distance_distribution(vanlint)
array([ 1., 12., 66., 46.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

```
