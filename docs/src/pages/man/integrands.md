# Integrands

## User-defined

For integrands that can be evaluated by Wannier interpolation, the following
data are necessary to define an integrand:
- the integrand evaluator
- a Fourier series
- additional parameters for the evaluator
Consider implementing custom integrands using the generic template type
[`AutoBZ.Applications.WannierIntegrand`](@ref) that is compatible with all of
the adaptive and equispace integration routines.

!!! tip "Optimizing equispace integration"
    Unlike for adaptive integration, the caller is responsible for passing
    pre-computed grid values to the equispace integration routines, which is
    explained in the documentation for [Equispace integration](@ref) and
    [`AutoBZ.Applications.equispace_pre_eval`](@ref).

!!! warning "Mixing adaptive and equispace integrals"
    While it is possible to perform an integral where some variables are
    integrated adaptively and others are integrated uniformly, this guide will
    not explain how to do this. However, an example implementation of this is 
    [`AutoBZ.Applications.AutoEquispaceKineticIntegrand`](@ref).


### Types

```@docs
AutoBZ.Applications.WannierIntegrand
```

### Methods

```@docs
AutoBZ.Applications.pre_eval_contract(::AutoBZ.Applications.WannierIntegrand,::Any,::Any)
```

## Pre-defined

### Types

```@docs
AutoBZ.Applications.DOSIntegrand
AutoBZ.Applications.TransportIntegrand
AutoBZ.Applications.KineticIntegrand
AutoBZ.Applications.EquispaceKineticIntegrand
AutoBZ.Applications.AutoEquispaceKineticIntegrand
```

### Functions

```@docs
AutoBZ.Applications.dos_integrand
AutoBZ.Applications.spectral_function
AutoBZ.Applications.band_velocities
AutoBZ.Applications.transport_integrand
AutoBZ.Applications.kinetic_integrand
AutoBZ.Applications.fermi
AutoBZ.Applications.fermi′
AutoBZ.Applications.fermi_window
AutoBZ.Applications.cosh_ratio
AutoBZ.Applications.EXP_P1_SMALL_X
```