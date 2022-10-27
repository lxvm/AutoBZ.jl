# Fourier series

Wannier-interpolated Hamiltonians are represented by Fourier series with a
modest number of Fourier coefficients. The goal of this page of documentation is
to describe the features, interface, and conventions of Fourier series
evaluation as implemented by this library.

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
AutoBZ.Applications.BandEnergyVelocity
```

## Methods

```@docs
AutoBZ.Applications.contract(::AutoBZ.Applications.AbstractFourierSeries)
```