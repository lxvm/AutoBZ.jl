# BZ visualization

In addition to visualizing band diagrams, which focus on high-symmetry paths of
the BZ, it can be useful to identify contributions from a full 3d view of the
BZ. The examples below show how to do so using AutoBZ

## Spectral function

```@example viz
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

using LinearAlgebra
ω = 0.0
μ = -0.6
η = 0.05 # broadening
Σ = EtaSelfEnergy(η)
bz = load_bz(FBZ(3), Diagonal(collect(AutoBZ.period(H))))
solver = DOSSolver(Σ, H, bz, PTR(npt=50); ω, μ)
f = solver.f
ksolver = init(f.prob, f.alg; f.kwargs...)
kpts = range(-0.5, 0.5; length=50)
kvals = map(Iterators.product(kpts, kpts, kpts[1:26])) do k
    h_k = f.s(k)
    f.update!(ksolver, k, h_k, solver.p)
    sol = solve!(ksolver)
    f.postsolve(sol, k, h_k, solver.p)
end
kvals_density = kvals ./ maximum(kvals)

using GLMakie
v = volume(kvals_density; algorithm=:iso, isovalue=1.0, isorange=0.9, 
colormap=cgrad([:teal, :teal],10))
save("dos_k.png", v); nothing # hide
```

![DOS BZ visualization](dos_k.png)


## Transport function

```@example viz
β = 10.0
μ = -0.6
bz = load_bz(FBZ(3), Diagonal(collect(AutoBZ.period(H))))
hv = GradientVelocityInterp(H, bz.A; gauge=Hamiltonian())
solver = TransportFunctionSolver(hv, bz, PTR(npt=50); β, μ)
f = solver.f
kpts = range(-0.5, 0.5; length=50)
kvals = map(Iterators.product(kpts, kpts, kpts[1:26])) do k
    h_k = f.s(k)
    f.f(k, h_k, solver.p)
end
kvals_density = -1 .* real.(tr.(kvals)) ./ maximum(norm, kvals)

v = volume(kvals_density; algorithm=:iso, isovalue=0.5, isorange=0.05, colormap=cgrad([:teal, :teal],10))
save("tf_k.png", v); nothing # hide
```

![transport function BZ visualization](tf_k.png)

## Conductivity

```@example viz
Ω = 0.4
solver = OpticalConductivitySolver(hv, bz, PTR(npt=50), Σ, QuadGKJL(); μ, β, Ω, abstol=1e-3, reltol=1e-3)
ksolver = init(solver.f.prob, solver.f.alg; solver.f.kwargs...)

kvals = map(Iterators.product(kpts, kpts, kpts[1:26])) do k
    hvk = hv(k)
    solver.f.update!(ksolver, k, hvk, solver.p)
    sol = solve!(ksolver)
    solver.f.postsolve(sol, k, hvk, solver.p)
end
kvals_density = real.(tr.(kvals)) ./ maximum(real.(tr.(kvals)))

v = volume(kvals_density; algorithm=:iso, isovalue=1.0, isorange=0.5,
colormap=cgrad([:teal, :teal],10))
save("oc_k.png", v); nothing # hide
```

![optical conductivity BZ visualization](oc_k.png)
