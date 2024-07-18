# Electron density

The [electron density](https://en.wikipedia.org/wiki/Electron_density) describes
the number of electrons in a system. It can be calculated by integrating the DOS
times a Fermi distribution over all frequencies. Often calculations of the
density are needed to ensure charge self-consistency.

## Model calculation

For this tutorial and the optical conductivity tutorial we define a simple
tight-binding model based on ``t_{2g}`` orbitals with a nearest neighbor
intraband hopping and a next-nearest neighbor interband hopping.
```@example chempot
using StaticArrays
using OffsetArrays
using AutoBZ
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
H = HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(H, period=2pi)))
```
With this Hamiltonian we can define an [`AutoBZ.ElectronDensitySolver`](@ref)
and a solver that computes the electron density at a given temperature and
scattering rate.
```@example chempot
using LinearAlgebra
bz = load_bz(CubicSymIBZ(), Diagonal(collect(AutoBZ.period(H))))
η = 0.1 # eV
β = 10.0 # 1/eV
Σ = EtaSelfEnergy(η)
atol=1e-3
rtol=0.0
solver = ElectronDensitySolver(H, bz, PTR(npt=50), Σ, (-Inf, Inf), QuadGKJL(); β, abstol=atol/nsyms(bz), reltol=rtol)
```
Here, we have chosen to the order of integration to compute a frequency integral
for each ``\bm{k}`` point. We can compute the density over a range of chemical
potentials and account for the normalization of the integral
```@example chempot
ENV["GKSwstype"] = "100" # hide
using Plots
freqs = range(-2, 2, length=100)
plot(freqs, μ -> (AutoBZ.update_density!(solver; β, μ); solve!(solver).value*2/det(bz.B)), title="Two hopping model", xguide="μ", yguide="Electron filling", label="η=$η, β=$β")
savefig("number_density.png"); nothing # hide
```

![model electron density](number_density.png)


## Chemical potential finder

Using the electron density solver above, we can easily create a chemical
potential finder from
[SimpleNonlinearSolve.jl](https://github.com/SciML/SimpleNonlinearSolve.jl)
root-finding algorithms.

```@example chempot
using SimpleNonlinearSolve
number_density(μ, (solver, n_sp, ν, V, β)) = (AutoBZ.update_density!(solver; μ, β); ν - solve!(solver).value*n_sp/V)
interval = (-1.0, 1.0)
prob = IntervalNonlinearProblem(number_density, interval, (solver, 2, 1.0, det(bz.B), β))
solve(prob, ITP())
```
In this way, it is possible to study the temperature dependence of the chemical
potential.
```@example chempot
ENV["GKSwstype"] = "100" # hide
using Plots
temps = range(100, 300, length=10)
f = T -> solve(remake(prob, u0=interval, p=(solver, 2, 1.0, det(bz.B), inv(8.617333262e-5*T))), ITP()).u
plot(temps, f, title="Two hopping model", xguide="T", yguide="μ", label="η=$η")
savefig("chempot.png"); nothing # hide
```

![model chemical potential](chempot.png)

It is important to note that if you are using a frequency-dependent self energy
that you should check the total number of electrons in the system is as
expected. This can be done by integrating the DOS over all frequencies at a
given chemical potential.