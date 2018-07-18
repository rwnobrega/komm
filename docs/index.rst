:tocdepth: 2

Komm
====

Welcome to **Komm**'s documentation page!

**Komm** is an open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems. This project is inspired by---but is not meant to be compatible with---the MATLAB® `Communications System Toolbox™ <https://www.mathworks.com/help/comm/>`_. Other sources of inspiration include `GNU Radio <https://gnuradio.org/>`_, `CommPy <http://veeresht.info/CommPy/>`_, and `SageMath <https://www.sagemath.org/>`_. **Komm** is licensed under the `GNU General Public License v3.0 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.

For installation instructions and source code, please check the project's `development page at GitHub <https://github.com/rwnobrega/komm>`_.

This software is still under development.

.. currentmodule:: komm

Algebra
-------

.. autosummary::
    :toctree:
    :nosignatures:

    BinaryPolynomial
    BinaryPolynomialFraction
    BinaryFiniteExtensionField

Channels
--------

.. autosummary::
    :toctree:
    :nosignatures:

    AWGNChannel
    DiscreteMemorylessChannel
    BinarySymmetricChannel
    BinaryErasureChannel

Error control
-------------

Block coding
~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    BlockCode
    HammingCode
    SimplexCode
    GolayCode
    RepetitionCode
    SingleParityCheckCode
    ReedMullerCode
    CyclicCode
    BCHCode

Convolutional coding
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    ConvolutionalCode
    ConvolutionalStreamEncoder
    ConvolutionalStreamDecoder
    TerminatedConvolutionalCode

Finite-state machine
--------------------

.. autosummary::
    :toctree:
    :nosignatures:

    FiniteStateMachine

Modulation
----------

Real modulation schemes
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    RealModulation
    PAModulation

Complex modulation schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    ComplexModulation
    ASKModulation
    PSKModulation
    APSKModulation
    QAModulation

Pulse formatting
----------------

Pulses
~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    RectangularPulse
    ManchesterPulse
    SincPulse
    RaisedCosinePulse
    RootRaisedCosinePulse
    GaussianPulse

Filtering
~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    TransmitFilter
    ReceiveFilter

Sequences
---------

Binary sequences
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    BinarySequence
    BarkerSequence
    WalshHadamardSequence
    LFSRSequence
    GoldSequence
    KasamiSequence

Other sequences
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    ZadoffChuSequence

Source coding
-------------

Lossless coding
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
    :nosignatures:

    PrefixCode
    HuffmanCode
    TunstallCode

Utilities
---------

.. autosummary::
    :toctree:
    :nosignatures:

    binlist2int
    int2binlist
    pack
    unpack
    qfunc
    qfuncinv
    entropy
    mutual_information

Indices
-------

* :ref:`glossary`
* :ref:`references`
* :ref:`genindex`
