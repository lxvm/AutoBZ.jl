# Optical conductivity

The [optical conductivity](https://en.wikipedia.org/wiki/Optical_conductivity)
is a response function that describes the electrical current response of a
material to an incident electromagnetic field. AutoBZ.jl currently implements
the longitudinal conductivity, which is the symmetric part of the conductivity
tensor.

## Model conductivity

For this tutorial and the electron density tutorial we define a simple
tight-binding model based on ``t_{2g}`` orbitals with a nearest neighbor
intraband hopping and a next-nearest neighbor interband hopping.
```@example oc
using StaticArrays
using OffsetArrays
using AutoBZ
using LinearAlgebra
bz = load_bz(CubicSymIBZ(), Diagonal(ones(3)))
H = OffsetArray(zeros(SMatrix{3,3,Float64,9}, 3,3,3), -1:1, -1:1, -1:1)
# intraband hoppings
t = -0.25 # nearest-neighbor hopping
H[ 1, 0, 0] = H[-1, 0, 0] =  [ 0; 0; 0;; 0; t; 0;; 0; 0; t]
H[ 0, 1, 0] = H[ 0,-1, 0] =  [ t; 0; 0;; 0; 0; 0;; 0; 0; t]
H[ 0, 0, 1] = H[ 0, 0,-1] =  [ t; 0; 0;; 0; t; 0;; 0; 0; 0]
# interband hoppings
t′ = 0.05 # next-nearest neighbor hopping
H[ 0, 1, 1] = H[ 0,-1,-1] =  [ 0; 0; 0;; 0; 0;t′;; 0;t′; 0]
H[ 0, 1,-1] = H[ 0,-1, 1] = -[ 0; 0; 0;; 0; 0;t′;; 0;t′; 0]
H[ 1, 0, 1] = H[-1, 0,-1] =  [ 0; 0;t′;; 0; 0; 0;;t′; 0; 0]
H[ 1, 0,-1] = H[-1, 0, 1] = -[ 0; 0;t′;; 0; 0; 0;;t′; 0; 0]
H[ 1, 1, 0] = H[-1,-1, 0] =  [ 0;t′; 0;;t′; 0; 0;; 0; 0; 0]
H[ 1,-1, 0] = H[-1, 1, 0] = -[ 0;t′; 0;;t′; 0; 0;; 0; 0; 0]
hv = GradientVelocityInterp(HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(H, period=2pi))), bz.A)
```
The optical conductivity requires the velocity operators in addition to the
Hamiltonian in order to compute the current-current correlations. To load a
Hamiltonian and velocities from Wannier90 data, see
[`AutoBZ.load_wannier90_data`](@ref). For integrating the optical conductivity,
we construct an [`AutoBZ.OpticalConductivitySolver`](@ref) and a solver for
the BZ integral
```@example oc
η = 0.1 # eV
μ = -0.669607319787773 # eV
β = 10.0 # 1/eV
Σ = EtaSelfEnergy(η)
atol=1e-3
rtol=0.0
solver = OpticalConductivitySolver(hv, bz, PTR(npt=50), Σ, QuadGKJL(); β, Ω=0.0, μ, abstol=atol/nsyms(bz), reltol=rtol);
nothing # hide
```
Then we can evaluate the frequency dependence of the conductivity and plot
particular matrix elements.
```@example oc
ENV["GKSwstype"] = "100" # hide
using Plots
freqs = range(0, 1, length=100)
plot(freqs, Ω -> (AutoBZ.update_oc!(solver; β, Ω, μ); real(solve!(solver).value[1,1])), title="Two hopping model", xguide="Ω", yguide="σₓₓ (a.u.)", label="η=$η, β=$β")
savefig("conductivity.png"); nothing # hide
```

![model conductivity](conductivity.png)


## Kinetic coefficients

A generalization of the optical conductivity is the
[`AutoBZ.KineticCoefficientSolver`](@ref), which enables the calculation of
additional transport properties. For example, we can compute the Seebeck
coefficient as a function of temperature
```@example oc
solver_1 = KineticCoefficientSolver(hv, bz, PTR(npt=50), Σ, QuadGKJL(); n=1, Ω=0.0, β, μ, abstol=atol/nsyms(bz), reltol=rtol)
temps = range(100, 300, length=10)
f = T -> begin
    AutoBZ.update_oc!(solver; β=inv(8.617333262e-5*T), Ω=0.0, μ)
    AutoBZ.update_kc!(solver_1; β=inv(8.617333262e-5*T), Ω=0.0, μ, n=1)
    -real(solve!(solver_1).value[1,1]) / real(solve!(solver).value[1,1])
end
plot(temps, f, title="Two hopping model", xguide="T", yguide="κₓₓ (a.u.)", label="η=$η")
savefig("seebeck.png"); nothing # hide
```

![model Seebeck](seebeck.png)

The kinetic coefficients calculate the higher moments of the
[`AutoBZ.TransportDistributionSolver`](@ref) and are especially useful for
thermal properties of solids.

## Auxiliary integration

For very small scattering rates, i.e. ``\eta < 10`` meV, adaptive integration
algorithms are more efficient than uniform integration. However, they may suffer
from a peak missing problem that we address with a technique called auxiliary
integration.

```@example oc
using IteratedIntegration: AuxValue
η = 0.01 # eV
aux_atol = 1e-2
trG_auxfun(vs, G1, G2) = tr(G1) + tr(G2)
aux_solver = AuxOpticalConductivitySolver(trG_auxfun, hv, bz, IAI(AuxQuadGKJL()), Σ, AuxQuadGKJL(); Ω=0.0, μ, β, abstol=AuxValue(atol/η,aux_atol), reltol=rtol)
solve!(aux_solver).value.val
```

To summarize this method, we define a helper function, `trG_auxfun` that takes
the velocities and Green's functions as its arguments, and should evaluate a
quantity that is peaked in the same place as the conductivity integrand. Since
the Green's function is less singular than the conductivity integrand at optical
transitions, it is nicer to integrate adaptively and it helps the algorithm
locate all highly-localized peaks with minimal additional effort.