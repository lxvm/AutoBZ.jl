# Integration limits

## Interface

```@docs
IntegrationLimits
limits
box
symmetries
ndims
nsyms
```

Additionally, all `IntegrationLimits` must extend `Base.eltype` to return the
type which is the output of `limits`, which is the type of
coordinates in the domain.

## Types

```@docs
CubicLimits
CompositeLimits
```

## Routines

```@docs
vol
symmetrize
AutoBZ.discretize_equispace
```