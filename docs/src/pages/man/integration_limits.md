# Integration limits

## Interface

```@docs
AutoBZ.AutoBZCore.IntegrationLimits
AutoBZ.AutoBZCore.limits
AutoBZ.AutoBZCore.box
AutoBZ.AutoBZCore.symmetries
AutoBZ.AutoBZCore.ndims
AutoBZ.AutoBZCore.nsyms
```

Additionally, all `IntegrationLimits` must extend `Base.eltype` to return the
type which is the output of `limits`, which is the type of
coordinates in the domain.

## Types

```@docs
AutoBZ.AutoBZCore.CubicLimits
AutoBZ.AutoBZCore.CompositeLimits
```

## Routines

```@docs
AutoBZ.AutoBZCore.vol
AutoBZ.AutoBZCore.symmetrize
AutoBZ.AutoBZCore.discretize_equispace
```