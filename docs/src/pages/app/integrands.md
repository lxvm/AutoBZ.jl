# Integrands

AutoBZ.jl functionality for BZ integration is entirely expressed through
functions that are integrated using AutoBZCore.jl. As such, anyone can implement
their own BZ integrals in the same way that AutoBZ.jl implements them.

## User-defined

For integrands that can be evaluated by Wannier interpolation, the following
data are necessary to define an integrand:
- the integrand evaluator
- a Fourier series
- additional parameters for the evaluator
Consider implementing custom integrands using the generic template type
`AutoBZCore.FourierIntegralFunction` that is compatible with all of
the adaptive and equispace integration routines. Using this interface will
automatically evaluate the multi-dimensional Fourier series in an efficient
manner for each integration routines.

## Pre-defined

### Constructors

```@docs
AutoBZ.GlocSolver
AutoBZ.TrGlocSolver
AutoBZ.DOSSolver
AutoBZ.TransportFunctionSolver
AutoBZ.TransportDistributionSolver
AutoBZ.KineticCoefficientSolver
AutoBZ.OpticalConductivitySolver
AutoBZ.ElectronDensitySolver
AutoBZ.AuxTransportDistributionSolver
AutoBZ.AuxKineticCoefficientSolver
AutoBZ.AuxOpticalConductivitySolver
```

### Functions

```@docs
AutoBZ.fermi
AutoBZ.fermi′
AutoBZ.fermi_window
AutoBZ.fermi_window_limits
```