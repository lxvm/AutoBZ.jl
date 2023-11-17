# Fourier series

Wannier-interpolated Hamiltonians are represented by Fourier series with a
modest number of Fourier coefficients. The goal of this page of documentation is
to describe the features, interface, and conventions of Fourier series
evaluation as implemented by this library.

## Conventions

Fourier series represent functions as linear combinations of sinusoids whose
frequencies are integer multiples of a fundamental mode. In the band theory of
solids, the fundamental frequencies, or normal modes, for a Hamiltonian
correspond to linear combinations of real-space lattice vectors that generate a
Bravais lattice. The sections below define conventions for each of these linear
combinations.

### Lattice vectors
It is conventional to construct a reciprocal lattice ``\{\bm{b}_j\}`` from a
Bravais lattice ``\{\bm{a}_i\}`` such that 
```math
\bm{b}_j \cdot \bm{a}_i = 2\pi\delta_{ij}
```
Then we write the momentum space variable in the reciprocal lattice vector basis
and the position space variable in the real lattice vector basis
```math
\bm{k} = \sum_{j=1}^{d} k_j \bm{b}_j
\qquad
\bm{R} = \sum_{i=1}^{d} R_i \bm{a}_i
```
Without loss of generality, the ``k_j`` can be taken in the domain
``[0,1]`` and the ``R_i`` are integers.

Additionally, any coordinate transformations of ``\bm{k}`` from the
Cartesian basis to the reciprocal lattice basis only modify Brillouin zone
integrals by a multiplicative factor of the absolute value of the determinant of
the basis transformation.

### Series coefficients
Wannier-interpolated Hamiltonians are small matrices obtained by projecting the
ab-initio Hamiltonian onto a low energy subspace, a process called downfolding.
These Hamiltonians can be expressed by a Fourier series as in the sum below
```math
H(\bm{k}) = \sum_{\bm{R}} e^{i\bm{k}\cdot\bm{R}} H_{\bm{R}}
```
where the coefficients ``H_{\bm{R}}`` are the matrix-valued Fourier
coefficients. Truncating the sum over ``\bm{R}`` at a modest number of modes can
be done for Wannier Hamiltonians in the maximally-localized orbital basis, for
which ``H(\bm{k})`` is a smooth and periodic function and thus the truncation
error of its Fourier series converges super-algebraically with respect to the
number of modes.

### Hamiltonian recipe
In model systems, a Bloch Hamiltonian can often be written down analytically.
The recipe to write it as a Fourier series has two-steps
1. Identify the real and reciprocal Bravais lattices, ``\{\bm{a}_i\}`` and
   ``\{\bm{b}_j\}``, and rewrite all of the phase dependences of the Hamiltonian
   as ``\bm{k}\cdot\bm{R}`` with each vector in its corresponding basis, as
   explained above.
2. Factor the Hamiltonian into a linear combination of normal modes indexed by
   the distinct ``\bm{R}`` vectors. If the Hamiltonian is matrix-valued, this
   can be done one matrix element at a time.

## Interface

See
[FourierSeriesEvaluators.jl](https://github.com/lxvm/FourierSeriesEvaluators.jl)
for the `AbstractFourierSeries` interface, which allows evaluation of the series
with a function-like `f(x)` syntax

## Types

The concrete types listed below all implement the `AbstractFourierSeries`
interface and should cover most use cases.

```@docs
AutoBZ.AbstractWannierInterp
AutoBZ.AbstractGaugeInterp
AutoBZ.AbstractHamiltonianInterp
AutoBZ.HamiltonianInterp
AutoBZ.AbstractCoordInterp
AutoBZ.BerryConnectionInterp
AutoBZ.AbstractVelocityInterp
AutoBZ.GradientVelocityInterp
AutoBZ.CovariantVelocityInterp
AutoBZ.MassVelocityInterp
```

### Gauges
```@docs
AutoBZ.AbstractGauge
AutoBZ.Wannier
AutoBZ.Hamiltonian
```

### Coordinate systems
```@docs
AutoBZ.AbstractCoordinate
AutoBZ.Cartesian
AutoBZ.Lattice
```

### Velocity components
```@docs
Whole
Intra
Inter
```