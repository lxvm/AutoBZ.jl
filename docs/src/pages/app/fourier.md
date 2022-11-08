# Fourier series

Wannier-interpolated Hamiltonians are represented by Fourier series with a
modest number of Fourier coefficients. The goal of this page of documentation is
to describe the features, interface, and conventions of Fourier series
evaluation as implemented by this library.

!!! note "Representing Hamiltonians with Fourier series"
    Fourier series represent functions with sinusoids whose frequencies are
    integer multiples of a fundamental. In the band theory of solids, the
    fundamental frequencies for a Hamiltonian correspond to the real-space
    lattice vectors that generate a Bravais lattice, and so it is best to
    represent the momentum variable ``\vec{k}`` in the basis of the reciprocal
    lattice. Since it is conventional to construct a reciprocal lattice
    ``\{\vec{b}_j\}`` from a Bravais lattice ``\{\vec{a}_i\}`` such that
    ``\vec{b}_j \cdot \vec{a}_i = 2\pi\delta_{ij}``, the Fourier series in this
    library are defined with a phase factor scaled by ``2\pi``. Additionally,
    any coordinate transformations of ``\vec{k}`` from the Cartesian basis to
    the reciprocal lattice basis only modify Brillouin zone integrals by a
    multiplicative factor of the absolute value of the determinant of the basis
    transformation. To find a non-trivial example of representing a Hamiltonian
    in the reciprocal lattice basis, see the [Graphene example with
    `ManyOffsetsFourierSeries`](@ref).

## Interface

```@docs
AutoBZ.Applications.AbstractFourierSeries
AutoBZ.Applications.period
AutoBZ.Applications.contract
AutoBZ.Applications.value
```

Additonally, concrete subtypes of `AbstractFourierSeries` must have an element
type, which they can do by extending `Base.eltype` with a method. For example,
if a type `MyFourierSeries <: AbstractFourierSeries` always returns `ComplexF64`
outputs, then the correct `eltype` method to define would be:
```julia
Base.eltype(::Type{MyFourierSeries}) = ComplexF64
```
The type returned should correspond to the vector space ``V`` of the output
space of the Fourier series, i.e. the output of `value` should be of this
type. For good performance, the `eltype` should be a concrete type and should be
inferrable.

With the above implemented, several methods which define functors for
`AbstractFourierSeries` allow the user (and integration routines) to evaluate
the type like a function with the `f(x)` syntax.

## Types

The concrete types listed below all implement the `AbstractFourierSeries`
interface and should cover most use cases.

```@docs
AutoBZ.Applications.FourierSeries
AutoBZ.Applications.FourierSeriesDerivative
AutoBZ.Applications.OffsetFourierSeries
AutoBZ.Applications.ManyFourierSeries
AutoBZ.Applications.ManyOffsetsFourierSeries
AutoBZ.Applications.BandEnergyVelocity
```

## Methods

```@docs
AutoBZ.Applications.contract(::AutoBZ.Applications.AbstractFourierSeries)
AutoBZ.Applications.band_velocities
```