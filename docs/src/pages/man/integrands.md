# Integrands

## User-defined

For integrands that can be evaluated by Wannier interpolation, the following
data are necessary to define an integrand:
- the integrand evaluator
- a Fourier series
- additional parameters for the evaluator
Consider implementing custom integrands using the generic template type
[`AutoBZ.WannierIntegrand`](@ref) that is compatible with all of
the adaptive and equispace integration routines.

!!! tip "Optimizing equispace integration"
    Unlike for adaptive integration, the caller is responsible for passing
    pre-computed grid values to the equispace integration routines, which is
    explained in the documentation for [Equispace integration](@ref) and
    [`AutoBZ.equispace_pre_eval`](@ref).

!!! warning "Mixing adaptive and equispace integrals"
    While it is possible to perform an integral where some variables are
    integrated adaptively and others are integrated uniformly, this guide will
    not explain how to do this. However, an example implementation of this is 
    [`AutoBZ.AutoEquispaceKineticIntegrand`](@ref).


### Types

```@docs
AutoBZ.WannierIntegrand
```

### Methods

```@docs
AutoBZ.pre_eval_contract(::AutoBZ.WannierIntegrand,::Any,::Any)
```

## Pre-defined

### Types

```@docs
AutoBZ.DOSIntegrand
AutoBZ.TransportIntegrand
AutoBZ.KineticIntegrand
AutoBZ.EquispaceKineticIntegrand
AutoBZ.AutoEquispaceKineticIntegrand
```

### Functions

```@docs
AutoBZ.dos_integrand
AutoBZ.spectral_function
AutoBZ.band_velocities
AutoBZ.transport_integrand
AutoBZ.kinetic_integrand
AutoBZ.fermi
AutoBZ.fermi′
AutoBZ.fermi_window
AutoBZ.cosh_ratio
AutoBZ.EXP_P1_SMALL_X
```