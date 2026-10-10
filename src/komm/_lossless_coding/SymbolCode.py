from functools import cached_property

from .._util.validators import validate_integer
from .util import Word


class SymbolCode:
    r"""
    General symbol code. A *symbol code* with *source alphabet* $\mathcal{X}$ and *target alphabet* $\mathcal{Y}$ is defined by an *encoding mapping* $\Enc: \mathcal{X} \to \mathcal{Y}^+$, where the co-domain is the set of all finite-length, non-empty tuples with entries in $\mathcal{Y}$. The elements in the image of $\Enc$ are called *codewords*. For more details, see <cite>MacK03, Ch. 5</cite>.

    A symbol code is equivalent to a [fixed-to-variable length code](/ref/FixedToVariableCode) with source block size $k = 1$, in which each source symbol $x$ corresponds to the $1$-tuple $(x)$.

    Note:
        Here, for simplicity, we assume that the source alphabet is $\mathcal{X} = [0 : |\mathcal{X}|)$ and the target alphabet is $\mathcal{Y} = [0 : |\mathcal{Y}|)$, where $|\mathcal{X}| \geq 2$ and $|\mathcal{Y}| \geq 2$ are called the *source cardinality* and *target cardinality*, respectively.
    """

    def __init__(
        self,
        source_cardinality: int,
        target_cardinality: int,
        enc_mapping: dict[int, Word],
    ) -> None:
        calX = validate_integer(source_cardinality, low=2)
        calY = validate_integer(target_cardinality, low=2)
        domain, codomain = enc_mapping.keys(), enc_mapping.values()
        if not set(domain) == set(range(calX)):
            raise ValueError("'enc_mapping': invalid domain")
        if not all(len(word) > 0 for word in codomain):
            raise ValueError("'enc_mapping': codewords must be non-empty")
        if not all(all(0 <= y < calY for y in word) for word in codomain):
            raise ValueError("'enc_mapping': invalid co-domain")
        self._source_cardinality = calX
        self._target_cardinality = calY
        self._enc_mapping = {x: enc_mapping[x] for x in range(calX)}

    def __repr__(self) -> str:
        args = ", ".join([
            f"source_cardinality={self.source_cardinality}",
            f"target_cardinality={self.target_cardinality}",
            f"enc_mapping={self.enc_mapping}",
        ])
        return f"{self.__class__.__name__}({args})"

    @cached_property
    def source_cardinality(self) -> int:
        r"""
        The source cardinality $|\mathcal{X}|$ of the code. It is the number of symbols in the source alphabet.

        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.source_cardinality
            3
        """
        return self._source_cardinality

    @cached_property
    def target_cardinality(self) -> int:
        r"""
        The target cardinality $|\mathcal{Y}|$ of the code. It is the number of symbols in the target alphabet.

        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.target_cardinality
            2
        """
        return self._target_cardinality

    @cached_property
    def enc_mapping(self) -> dict[int, Word]:
        r"""
        The encoding mapping $\Enc$ of the code. It is a dictionary of length $|\mathcal{X}|$ whose keys are all the integers in $\mathcal{X}$ and whose values are the corresponding codewords.

        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.enc_mapping
            {0: (0,),
             1: (1, 0),
             2: (1, 1)}
        """
        return self._enc_mapping

    @cached_property
    def codewords(self) -> list[Word]:
        r"""
        The codewords of the code. They correspond to the image of the encoding mapping $\Enc$: the tuple in position $x$ is equal to $\Enc(x)$.

        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.codewords
            [(0,), (1, 0), (1, 1)]
        """
        return list(self.enc_mapping.values())

    @cached_property
    def lengths(self) -> list[int]:
        r"""
        The codeword lengths of the code. The integer in position $x$ is equal to $\ell(x)$, the length of the codeword $\Enc(x)$.

        Examples:
            >>> code = komm.SymbolCode(3, 2, {0: (0,), 1: (1, 0), 2: (1, 1)})
            >>> code.lengths
            [1, 2, 2]
        """
        return [len(y) for y in self.codewords]
