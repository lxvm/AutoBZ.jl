# Integrands

AutoBZ.jl defines integrands to compute various physical observables, including
the density of states, electronic density, and optical conductivity. To define
new observables, visit
[AutoBZCore.jl](https://lxvm.github.io/AutoBZCore.jl/dev/) for general-purpose
interfaces to define integrals.

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