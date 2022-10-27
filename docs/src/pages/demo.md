# Demos

## DOS of the integer lattice tight-binding model

To demonstrate setting up a DOS calculation with AutoBZ, we consider a
tight-binding model on the ``n``-dimensional integer lattice with lattice
constant ``a`` and hopping strength ``t>0``:
```math
H = -t \sum_{i \in Z^n} \sum_{j=1}^n \ket{i}\bra{i+\hat{j}} + \ket{i+\hat{j}}\bra{i}
```
Solving this model by employing Bloch's theorem yields the following band
```math
H(k_1, \ldots, k_n) = -t(\cos(k_1 a) + \cdots + \cos(k_n a))
```
We shall input this Hamiltonian by constructing the equivalent Fourier series
```julia
using StaticArrays
using OffsetArrays

using AutoBZ
using AutoBZ.Applications

n = 3 # arbitrary positive integer
a = fill(1.0, SVector{n})
ax = repeat([-1:1], n)
C = zeros(SMatrix{1,1,ComplexF64,1}, ntuple(_ -> 3, n))
for i in 1:n, j in (-1, 1)
    C[CartesianIndex(ntuple(k -> k == i ? 2+j : 2, n))] = SMatrix{1,1,ComplexF64,1}(0.5)
end
H = FourierSeries(OffsetArray(C, ax...), a)
```
Then we can define the integration problem to compute DOS
```julia
ω = 1.0*n # frequency
η = 0.1 # broadening
μ = 0.0 # chemical potential
Σ = EtaEnergy(η) # self energy
D = DOSIntegrand(H, ω, Σ, μ) # integrand

# construct IBZ integration limits
c = CubicLimits(H.period)
t = TetrahedralLimits(c)

# set error tolerances
atol = 1e-3
rtol = 0.0

iterated_integration(D, t; callback=contract, atol=atol, rtol=rtol)
```
You will find a working example of this model in the `DOS_example.jl` demo that
computes DOS over a range of frequencies for this model

## Custom integrand

For integrands that can be evaluated by Wannier interpolation, the following
data are necessary to define an integrand:
- the integrand evaluator
- a Fourier series
- additional parameters
Consider implementing custom integrands using this generic template with a few
associated methods
```julia
struct WannierIntegrand{TF,TS<:AbstractFourierSeries,TP}
    f::TF
    s::TS
    p::TP
end
contract(w::WannierIntegrand, x) = WannierIntegrand(w.f, contract(w.s, x), p)
(w::WannierIntegrand)(x::SVector{1}) = w(only(x))
(w::WannierIntegrand)(x::Number) = w.f(w.s(x), w.p...)
```