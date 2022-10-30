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
Then we can define the integration problem to compute DOS, defined by the
integral
```math
\operatorname{DOS}(\omega) = \int_{\text{BZ}} d\vec{k} \operatorname{Tr}[\Im\{\omega+\mu-H(\vec{k})+i\eta\}]
```
where ``\mu`` is the chemical potential and ``\eta`` is a constant scattering rate.
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
computes DOS over a range of frequencies for this model.

## Custom integrands

For integrands that can be evaluated by Wannier interpolation, the following
data are necessary to define an integrand:
- the integrand evaluator
- a Fourier series
- additional parameters for the evaluator Consider implementing custom
integrands using the generic template type
[`AutoBZ.Applications.WannierIntegrand`](@ref), that is compatible with all of
the adaptive and equispace integration routines. For example, we can replicate
the preceding tight-binding example by defining an integrand with the custom
integrand type
```julia
using LinearAlgebra
dos(H_k::AbstractMatrix, ω, μ, η) = -tr(imag(inv(complex(ω+μ, η)*I-H_k)))/pi
D = WannierIntegrand(dos, H, (ω, μ, η))
```

!!! tip "Optimizing equispace integration"
    Unlike for adaptive integration, the caller is responsible for passing
    pre-computed grid values to the equispace integration routines, which is
    explained in the documentation for [Equispace integration](@ref) and
    [`AutoBZ.Applications.pre_eval_contract`](@ref).

!!! warning "Mixing adaptive and equispace integrals"
    While it is possible to perform an integral where some variables are
    integrated adaptively and others are integrated uniformly, this guide will
    not explain how to do this. However, an example implementation of this is 
    [`AutoBZ.Applications.AutoEquispaceOCIntegrand`](@ref).

## Graphene example with `ManyOffsetsFourierSeries`

Let's study an example motivated by graphene whose Hamiltonian is given by a
tight-binding model on the hexagonal lattice with lattice constant ``a`` and
hopping amplitude ``t``. Applying Bloch's theorem to each triangular sublattice
brings the Hamiltonian into block-diagonal form, where each block is of the form
```math
-t
\begin{pmatrix}
0 & f(k)
\\ f^*(k) & 0
\end{pmatrix}
```
where ``f(k) = e^{ik\cdot\delta_1} + e^{ik\cdot\delta_2} + e^{ik\cdot\delta_3}``
and ``\delta_1 = a\hat{x}, \delta_2 = a(-1/2\hat{x}+\sqrt{3}/2\hat{y}), \delta_3
= a(-1/2\hat{x}-\sqrt{3}/2\hat{y})``. To exactly construct this Fourier series,
we will have to rotate basis so that these vectors are precisely integer linear
combinations of the new lattice vectors. Note that by defining ``\hat{a}_1 =
(\delta_1 - \delta_3)/3a = (\hat{x} + 1/\sqrt{3}\hat{y})/2, \hat{a}_2 =
(\delta_1 - \delta_2)/3a = (\hat{x} - 1/\sqrt{3}\hat{y})/2`` we can write
``\delta_1 = a(\hat{a}_1 + \hat{a}_2), \delta_2 = a(\hat{a}_1 - 2\hat{a}_2),
\delta_3 = a(-2\hat{a}_1 + \hat{a}_2)``. Therefore our coordinate transformation
matrix, ``T`` from Cartesian coordinates to the triangular lattice,
``\{\vec{a}_i = 3a\hat{a}_i\}``, is
```math
T = \frac{1}{2}
\begin{pmatrix}
1 & 1/\sqrt{3}
\\ 1 & -1/\sqrt{3}
\end{pmatrix}
\qquad
T^{-1} =
\begin{pmatrix}
1 & 1
\\ \sqrt{3} & -\sqrt{3}
\end{pmatrix}
```
and note ``|\operatorname{det}(T)| = 1/2\sqrt{3}``. Now the corresponding
reciprocal lattice vectors are constructed by the relation ``\hat{b}_i =
\epsilon_{ij} (\hat{z} \times \hat{a}_j)`` and rescaling so that ``\hat{b}_i
\cdot \hat{a}_j = 2\pi\delta_{ij}``. This yields ``\hat{b}_1 =
2\pi(\hat{x}+\sqrt{3}\hat{y}) = 4\pi(\hat{a}_1 - 2\hat{a_2}), \hat{b}_2 =
2\pi(\hat{x}-\sqrt{3}\hat{y}) = 4\pi(2\hat{a}_1 - \hat{a_2})``. We would now
interpret ``k`` in this basis, and could also use ``T`` to map from the
Cartesian basis to it. Also observe that if ``a`` is the lattice
constant of the hexagonal lattice, then ``\sqrt{3}a`` is the lattice constant of
the triangular lattice, and ``2\pi/\sqrt{3}a`` is the lattice constant of the
reciprocal lattice. However, we will have to rescale integrals by factors of
``|\operatorname{det}{T}|`` because of our coordinate transformations.

Having chosen this suitable basis for $k$ and $x$, we can now express the
$k$-dependence of the block Hamiltonian as
```math
f(k) = e^{ik\cdot\delta_1} + e^{ik\cdot\delta_2} + e^{ik\cdot\delta_3}
= e^{iak\cdot(\hat{a}_1 + \hat{a}_2)} + e^{iak\cdot(\hat{a}_1 - 2\hat{a}_2)} + e^{iak\cdot(-2\hat{a}_1 + \hat{a}_2)}
```
which is amenable to a Fourier series representation.

Suppose that the integral we want to calculate is
```math
g(\vec{q}) = \int_{\text{BZ}} dk_x dk_y \frac{\lambda(\xi(\vec{k})) - \lambda(\xi(\vec{k}-\vec{q}))}{\xi(\vec{k}) - \xi(\vec{k}-\vec{q})}
```
where ``\xi(\vec{k}) = \operatorname{det}(H(\vec{k}))`` and ``\lambda(\omega) =
\partial_T f(\omega)`` is the temperature derivative of the Fermi distribution.
Since the integrand requires evaluation of the Hamiltonian at various
``k``-points simultaneously, the
[`AutoBZ.Applications.ManyOffsetsFourierSeries`](@ref) type can be used to do
this. Moreover, `AutoBZ.Applications` has functions to evaluate Fermi functions
and their derivatives. Putting everything together leads us to the code example
below
```julia
using StaticArrays
using OffsetArrays

using AutoBZ
using AutoBZ.Applications

a = 1.0
C = OffsetArray(zeros(SMatrix{2,2,ComplexF64,4}, (5,5)), -2:2, -2:2)
C[1,1]   = C[1,-2] = C[-2,1] = [0 1; 0 0]
C[-1,-1] = C[-1,2] = C[2,-1] = [0 0; 1 0]
H = FourierSeries(C, 2*pi/a)

T = 100.0 # K
kB = 8.617333262e-5 # eV/K
q = rand(SVector{2,Float64}) # arbitrary
f = ManyOffsetsFourierSeries(H, q)

lambda(x, T, kB) = -AutoBZ.Applications.fermi′(inv(kB*T), x)/(kB*T^2)
integrand_(f, T, kB) = (lambda(det(f[1]), T, kB) - lambda(det(f[2]), T, kB))/(det(f[1])-det(f[2]))
integrand = WannierIntegrand(integrand_, f, (T, kB))

c = CubicLimits(H.period)

# set error tolerances
atol = 1e-3
rtol = 0.0

iterated_integration(integrand, c; callback=contract, atol=atol, rtol=rtol)
```
You will find a working example of this code in the `graphene.jl` demo that
calculates this integral for values of ``\vec{q}`` in the Brillouin zone.