# Integrands

## User-defined

For integrands that can be evaluated by Wannier interpolation, the following
data are necessary to define an integrand:
- the integrand evaluator
- a Fourier series
- additional parameters for the evaluator
Consider implementing custom integrands using the generic template type
[`AutoBZ.FourierIntegrand`](@ref) that is compatible with all of
the adaptive and equispace integration routines.

!!! tip "Optimizing equispace integration"
    Unlike for adaptive integration, the caller is responsible for passing
    pre-computed grid values to the equispace integration routines, which is
    explained in the documentation for [Equispace integration](@ref) and
    [`AutoBZ.equispace_rule`](@ref).

!!! warning "Mixing adaptive and equispace integrals"
    While it is possible to perform an integral where some variables are
    integrated adaptively and others are integrated uniformly, this guide will
    not explain how to do this. However, an example implementation of this is 
    [`AutoBZ.Jobs.KineticIntegrand`](@ref).

In fact, all of the BZ integrals in [`AutoBZ.Jobs`](@ref) are defined this way.
In addition, they define a type alias in order to have a special name.


## Pre-defined

### Types

```@docs
AutoBZ.Jobs.DOSIntegrand
AutoBZ.Jobs.SafeDOSIntegrand
AutoBZ.Jobs.ExperimentalDOSIntegrand
AutoBZ.Jobs.GlocIntegrand
AutoBZ.Jobs.DiagGlocIntegrand
AutoBZ.Jobs.TransportIntegrand
AutoBZ.Jobs.KineticIntegrand
AutoBZ.Jobs.ElectronCountIntegrand
```

### Functions

```@docs
AutoBZ.Jobs.dos_integrand
AutoBZ.Jobs.safedos_integrand
AutoBZ.Jobs.gloc_integrand
AutoBZ.Jobs.diaggloc_integrand
AutoBZ.Jobs.spectral_function
AutoBZ.Jobs.transport_integrand
AutoBZ.Jobs.kinetic_integrand
AutoBZ.Jobs.kinetic_frequency_integral
AutoBZ.Jobs.electron_count_integrand
AutoBZ.Jobs.electron_count_frequency_integral
AutoBZ.Jobs.band_velocities
AutoBZ.Jobs.fermi
AutoBZ.Jobs.fermi′
AutoBZ.Jobs.fermi_window
AutoBZ.Jobs.cosh_ratio
AutoBZ.Jobs.EXP_P1_SMALL_X
```