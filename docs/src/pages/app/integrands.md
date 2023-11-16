# Integrands

## User-defined

For integrands that can be evaluated by Wannier interpolation, the following
data are necessary to define an integrand:
- the integrand evaluator
- a Fourier series
- additional parameters for the evaluator
Consider implementing custom integrands using the generic template type
`AutoBZCore.FourierIntegrand` that is compatible with all of
the adaptive and equispace integration routines. Using this interface will
automatically optimize multi-dimensional Fourier series evaluations with the
various integration routines.

## Pre-defined

### Types

```@docs
AutoBZ.GlocIntegrand
AutoBZ.DiagGlocIntegrand
AutoBZ.TrGlocIntegrand
AutoBZ.DOSIntegrand
AutoBZ.TransportFunctionIntegrand
AutoBZ.TransportDistributionIntegrand
AutoBZ.KineticCoefficientIntegrand
AutoBZ.OpticalConductivityIntegrand
AutoBZ.ElectronDensityIntegrand
AutoBZ.AuxTransportDistributionIntegrand
AutoBZ.AuxKineticCoefficientIntegrand
AutoBZ.AuxOpticalConductivityIntegrand
```

### Functions

```@docs
AutoBZ.fermi
AutoBZ.fermi′
AutoBZ.fermi_window
AutoBZ.fermi_window_limits
```