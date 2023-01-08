# Integration limits

## Interface

```@docs
IntegrationLimits
lower
upper
box
symmetries
ndims
nsyms
```

Additionally, all `IntegrationLimits` must extend `Base.eltype` to return the
type which is the output of `lower` and `upper`, which is the type of
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