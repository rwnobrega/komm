:tocdepth: 3

Komm
====

Welcome to **Komm**'s documentation!

About
-----

**Komm** is an open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems.  This project is inspired by---but is not meant to be compatible with---the MATLAB® `Communications System Toolbox™ <https://www.mathworks.com/help/comm/>`_. Other sources of inspiration include `GNU Radio <https://gnuradio.org/>`_, `CommPy <http://veeresht.info/CommPy/>`_, and `SageMath <https://www.sagemath.org/>`_. **Komm** is licensed under the `GNU General Public License v3.0 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_.

This software is still under development.

Library
-------

.. currentmodule:: komm

Algebra
~~~~~~~

.. autosummary::
    :toctree: stubs
    :nosignatures:

    BinaryPolynomial
    BinaryFiniteExtensionField

Channels
~~~~~~~~

.. autosummary::
    :toctree: stubs
    :nosignatures:

    AWGNChannel
    DiscreteMemorylessChannel
    BinarySymmetricChannel
    BinaryErasureChannel

Error control
~~~~~~~~~~~~~

Block coding
^^^^^^^^^^^^

.. autosummary::
    :toctree: stubs
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
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: stubs
    :nosignatures:

    ConvolutionalCode

Modulation
~~~~~~~~~~

.. autosummary::
    :toctree: stubs
    :nosignatures:

    RealModulation
    PAModulation
    ComplexModulation
    ASKModulation
    PSKModulation
    QAModulation

Pulses
~~~~~~

.. autosummary::
    :toctree: stubs
    :nosignatures:

    Pulse
    RectangularNRZPulse
    RectangularRZPulse
    ManchesterPulse
    SincPulse
    RaisedCosinePulse
    RootRaisedCosinePulse
    GaussianPulse

Sequences
~~~~~~~~~

.. autosummary::
    :toctree: stubs
    :nosignatures:

    BarkerSequence
    WalshHadamardSequence
    LFSRSequence

Source coding
~~~~~~~~~~~~~

Lossless coding
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: stubs
    :nosignatures:

    SymbolCode
    HuffmanCode

Installation
------------

Before you start, make sure you have both `Python 3 <https://www.python.org/>`_ and `NumPy <https://www.numpy.org/>`_ installed.

Using pip
~~~~~~~~~

.. code-block:: bash

    # Comming soon!

From source
~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/rwnobrega/komm
    # Comming soon!

Indices
-------

* :ref:`glossary`
* :ref:`references`
* :ref:`genindex`
