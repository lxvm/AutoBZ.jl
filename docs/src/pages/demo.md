# Demos

To illustrate the principles and practice of using `AutoBZ.jl`, the tutorials
below show how to setup and compute BZ integrals for various observables of toy,
tight-binding models with the package. Additionally, scripts corresponding to
the tutorials are available in the `demos` folder of the repo.

## DOS of the integer lattice tight-binding model

In this tutorial, we consider a tight-binding model on the ``n``-dimensional
integer lattice with lattice constant ``a`` and hopping strength ``t>0`` given
by the following Hamiltonian:
```math
H = -t \sum_{i \in Z^n} \sum_{j=1}^n \ket{i}\bra{i+\hat{j}} + \ket{i+\hat{j}}\bra{i}
```
where ``\ket{i}`` represents the state at lattice site ``i\in Z^n`` and
``\hat{j}`` represents a vector of zeros except for a one at position ``j``.
We will compute the density of states (DOS) of this system, which as a function
of ``n`` shows the dimension-dependent behavior of Van-Hove singularities.

Employing Bloch's theorem, which for this problem implies ``\ket{i+\hat{j}} =
e^{i\bm{k}\cdot\hat{j}}\ket{i}``, yields the following band structure
```math
H(k_1, \ldots, k_n) = -t(\cos(k_1 a) + \cdots + \cos(k_n a))
```
We shall input this Hamiltonian to `AutoBZ` by constructing an equivalent
Fourier series, which boils down to writing this Hamiltonian in the form
``H(\bm{k}) = \sum_{\bm{R}} e^{i\bm{k}\cdot\bm{R}} H_{\bm{R}}``, where
``\bm{R}`` is an integer multi-index. To do this, we follow the [Hamiltonian
recipe](@ref). In the first step, we identify the real and reciprocal lattice
basis vectors as the Cartesian coordinate basis and then observe that the
``\bm{R}`` vectors with non-zero coefficients are exactly the nearest neighbor
vectors ``\{\pm\hat{j}\}_{j=1}^{n}``. In step two, we identify the coefficients
to be ``-t/2`` for all the terms by simply writing the cosines as complex
exponentials. Finally we fill the array of coefficients by taking each
``\bm{R}`` to be the array index of the corresponding coefficient ``H_{\bm{R}}``.
```julia
using OffsetArrays

using AutoBZ

n = 3 # arbitrary positive integer representing the number of k-space dimensions
a = 1.0 # lattice spacing
t = 1.0 # hopping amplitude
#=
construct the array of scalar coefficients and use an OffsetArray so that the
array indices correspond to the R integer multi-index of the Fourier series
=#
C = OffsetArray(zeros(ntuple(_ -> 3, n)), ntuple(_ -> -1:1, n)...)
for i in 1:n, j in (-1, 1)
    C[CartesianIndex(ntuple(k -> k == i ? j : 0, n))] = -0.5t
end
H = FourierSeries(C, 2pi/a)
```
Then we can define the integration problem to compute DOS, defined by the
integral
```math
\operatorname{DOS}(\omega) = -\frac{1}{\pi} \int_{\text{BZ}} d\bm{k}\ \operatorname{Im}\ \left[ (\hbar\omega+\mu-H(\bm{k})+i\eta)^{-1} \right]
```
where ``\omega`` is a frequency variable, ``\bm{k}`` is the reciprocal space
vector, ``\mu`` is the chemical potential and ``\eta`` is a constant scattering
rate. We implement our own user-defined integrand with the
[`AutoBZ.WannierIntegrand`](@ref) type:
```julia
ω = t*n # frequency at the band edge/Van-Hove singularity
ħ = 1.0 # reduced Planck's constant
η = 0.1 # broadening
dos_integrand(H_k, ω, η) = -imag(inv(ħ*ω - H_k + im*η))/pi # integrand evaluator
D = WannierIntegrand(dos_integrand, H, (ω, η)) # user-defined integrand
```
To compute the integral, we also need to provide the limits of integration, to
specify an error tolerance, and to call one of the integration routines
```julia
IBZ = TetrahedralLimits(period(H)) # Irreducible BZ for cubic symmetries is tetrahedron

atol = 1e-3 # absolute error tolerance requests the result to within ±atol

I, E = iterated_integration(D, IBZ; atol=atol)
```
The routine returns the estimate of the integral `I` and an error estimate `E`.

You will find a working example of this model in the `DOS_example.jl` demo that
computes DOS over a range of frequencies for this model.

## DOS of Graphene

In this tutorial, we will build the Fourier series corresponding to a
tight-binding model of graphene. This example is more complex in that the
lattice vectors are not orthogonal and that there are multiple bands.

The tight-binding model on the hexagonal lattice with lattice constant ``a`` and
hopping amplitude ``t``. Applying Bloch's theorem to each triangular sublattice
brings the Hamiltonian into block-diagonal form, where each block is of the form
```math
-t
\begin{pmatrix}
0 & f(\bm{k})
\\ f^*(\bm{k}) & 0
\end{pmatrix}
```
where ``f(k) = \sum_{j=1}^{3} e^{i\bm{k}\cdot\bm{\delta}_j}``
depends on the nearest-neighbor vectors
```math
\bm{\delta}_1 = a\hat{x}
\qquad
\bm{\delta}_2 = a(-1/2\hat{x}+\sqrt{3}/2\hat{y})
\qquad
\bm{\delta}_3 = a(-1/2\hat{x}-\sqrt{3}/2\hat{y})
```
To exactly construct this Fourier series, we begin with step one of the
[Hamiltonian recipe](@ref) identifying a basis of lattice vectors that forms a
Bravais lattice. We can choose these as the following triangular lattice vectors
```math
\bm{a}_1 = (\bm{\delta}_1 - \bm{\delta}_3)/3
\qquad
\bm{a}_2 = (\bm{\delta}_1 - \bm{\delta}_2)/3
```
such that in this basis we write
```math
\bm{\delta}_1 = \bm{a}_1 + \bm{a}_2
\qquad
\bm{\delta}_2 = \bm{a}_1 - 2\bm{a}_2
\qquad
\bm{\delta}_3 = -2\bm{a}_1 + \bm{a}_2
```
Now taking step two, we factor the Hamiltonian into different normal modes and
observe the ``\bm{R}`` vectors are just the pairs of integer coefficients in the
linear combination of Bravais lattice vectors for each exponential.
```math
(e^{i\bm{k}\cdot(\bm{a}_1 + \bm{a}_2)} + e^{i\bm{k}\cdot(\bm{a}_1 - 2\bm{a}_2)} + e^{i\bm{k}\cdot(-2\bm{a}_1 + \bm{a}_2)})
\begin{pmatrix}
0 & 0
\\ -t & 0
\end{pmatrix}
+ (e^{i\bm{k}\cdot(-\bm{a}_1 - \bm{a}_2)} + e^{i\bm{k}\cdot(-\bm{a}_1 + 2\bm{a}_2)} + e^{i\bm{k}\cdot(2\bm{a}_1 - \bm{a}_2)})
\begin{pmatrix}
0 & -t
\\ 0 & 0
\end{pmatrix}
```
This corresponds to the following Fourier series in `AutoBZ`
```julia
using StaticArrays
using OffsetArrays

using AutoBZ

a = 1.0 # length of Bravais lattice vectors
t = 1.0 # hopping amplitude
C = OffsetArray(zeros(SMatrix{2,2,ComplexF64,4}, (5,5)), -2:2, -2:2)
C[1,1]   = C[1,-2] = C[-2,1] = [0 -t; 0 0] # Define C[R] = H_R
C[-1,-1] = C[-1,2] = C[2,-1] = [0 0; -t 0]
H = FourierSeries(C, 2*pi/a)
```
The DOS integrand can be formulated as before, except it must also compute the
trace since this Hamiltonian is matrix-valued. Another option would be to use
the pre-defined [`AutoBZ.DOSIntegrand`](@ref).

## Graphene example with `ManyOffsetsFourierSeries`

Another Fourier series formulated from graphene is 
```math
\xi(\bm{k}) = \sum_{n=1}^{6} e^{i\bm{k}\cdot\bm{\delta}_{n}}
```
and by choosing a basis ``\bm{a}_1 = \hat{x}, \bm{a}_2 = (\hat{x} +
\sqrt{3}\hat{y})/2`` we express the displacement vectors as
```math
\bm{\delta}_1 = \bm{a}_1 = -\bm{\delta}_4
\qquad
\bm{\delta}_2 = \bm{a}_2 = -\bm{\delta}_5
\qquad
\bm{\delta}_3 = \bm{a}_2 - \bm{a}_1 = -\bm{\delta}_6
```
and we can construct the Fourier series in `AutoBZ` as
```julia
using OffsetArrays

using AutoBZ

C = OffsetArray(zeros(3, 3), -1:1, -1:1)
C[1,0] = C[-1,0] = C[0,1] = C[0,-1] = C[1,-1] = C[-1,1] = 1
ξ = FourierSeries(C, 2pi)
```
Now the new integral we want to calculate is
```math
g(\bm{q}) = \int_{\text{BZ}} d\bm{k} \frac{\lambda(\xi(\bm{k})) - \lambda(\xi(\bm{k}-\bm{q}))}{\xi(\bm{k}) - \xi(\bm{k}-\bm{q})}
```
where ``\lambda(\omega) = \partial_T f(\omega)`` is the temperature derivative
of the Fermi distribution. Since the integrand requires evaluation of the
Hamiltonian at various ``k``-points simultaneously, we can express this with a
[`AutoBZ.ManyOffsetsFourierSeries`](@ref). Moreover,
`AutoBZ` has functions to evaluate Fermi functions and their
derivatives. Putting these pieces together gives the following integrand definition
```julia
T = 100.0 # K
kB = 8.617333262e-5 # eV/K
q = rand(SVector{2,Float64}) # arbitrary q point to integrate
f = ManyOffsetsFourierSeries(ξ, q) # makes a new Fourier series from ξ offset by q

# define the function to integrate and wrap it in an integrand type
lambda(x, T, kB) = -x*fermi′(inv(kB*T)*x)/(kB*T^2)
function evaluate_integrand(f, T, kB)
    ξ_k, ξ_q = f
    if ξ_k == ξ_q
        # when the integrand is ill defined, return its limiting value ∂λ/∂ξ
        return lambda(ξ_k, T, kB)*(2*inv(kB*T)*fermi′(ξ_k*inv(kB*T))/fermi(ξ_k, inv(kB*T)) - inv(ξ_k) - inv(kB*T))
    else
        return (lambda(ξ_k, T, kB) - lambda(ξ_q, T, kB))/(ξ_k-ξ_q)
    end
end
integrand = WannierIntegrand(evaluate_integrand, f, (T, kB))
```
You will find a working example of this code in the `graphene.jl` demo that
calculates this integral for values of ``\bm{q}`` in the Brillouin zone.