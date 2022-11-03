export BandEnergyVelocity, BandEnergyBerryVelocity

"""
    BandEnergyVelocity(H::FourierSeries{N}, V::ManyFourierSeries{N}) where N
    BandEnergyVelocity(H::FourierSeries{N}) where N

The bottom constructor takes a Fourier series representing the Hamiltonian and
also evaluates the band velocities so that the return value after all the
dimensions are contracted is a tuple containing `(H, v₁, v₂, ..., vₙ)`. The band
velocities are defined by dipole operators ``\\nu_{\\alpha} = \\frac{1}{\\hbar}
\\partial_{k_{\\alpha}} H`` where ``k_{\\alpha}`` is one of three input
dimensions of ``H`` and ``\\hbar=1``. Effectively, this type evaluates `H` and
its gradient. Note that differentiation by ``k`` changes the units to have an
additional dimension of length and a factor of ``2\\pi``, so if ``H`` has
dimensions of energy, ``\\nu`` has dimensions of energy times length. The caller
is responsible for transforming the units of the velocity (i.e. ``\\hbar``) if
they want other units, which can usually be done as a post-processing step.
"""
struct BandEnergyVelocity{N,TH<:FourierSeries{N},TV<:ManyFourierSeries{N}} <: AbstractFourierSeries{N}
    H::TH
    V::TV
end
BandEnergyVelocity(H::FourierSeries{N}) where N = BandEnergyVelocity(H, ManyFourierSeries((), period(H)))
period(f::BandEnergyVelocity) = period(f.H)
function contract(f::BandEnergyVelocity{N}, x::Number) where N
    v = FourierSeriesDerivative(f.H, SVector(ntuple(n -> ifelse(n == N, 1, 0), Val{N}())))
    BandEnergyVelocity(contract(f.H, x), ManyFourierSeries((contract(v, x), contract(f.V, x).fs...), pop(period(f))))
end
value(f::BandEnergyVelocity{0}) = (value(f.H), value(f.V)...)
Base.eltype(::Type{BandEnergyVelocity{N,TH,TV}}) where {N,M,TH,TV<:ManyFourierSeries{N,<:NTuple{M}}} = NTuple{N+M+1,eltype(TH)}

"""
    BandEnergyBerryVelocity(H::BandEnergyVelocity{N}, A::ManyFourierSeries{N}) where N
    BandEnergyBerryVelocity(H::FourierSeries{N}, A::ManyFourierSeries{N,<:NTuple{N}}) where N

This constructor takes a `FourierSeries`, `H`, representing the Hamiltonian and
also a `ManyFourierSeries`, `A`, representing the gradient of the Berry
connection, and evaluates modified band velocities so that the return value
after all the dimensions are contracted is a tuple containing `(H, ṽ₁, ṽ₂,
..., ṽₙ)`. The modified band velocities are defined by
``\\tilde{\\nu}_{\\alpha} = \\frac{1}{\\hbar} \\partial_{k_{\\alpha}} H -
\frac{i}{\\hbar} [H,A_{\\alpha}]`` where ``k_{\\alpha}`` is one of three input
dimensions of ``H`` and ``\\hbar=1``. Effectively, this type evaluates the
Hamiltonian and its gradient modified by a commutator of the Hamiltonian with
the gradient of the Berry connection. Note that differentiation by ``k`` changes
the units to have an additional dimension of length and a factor of ``2\\pi``,
so if ``H`` has dimensions of energy, ``\\nu`` has dimensions of energy times
length. The caller is responsible for transforming the units of the velocity
(i.e. ``\\hbar``) if they want other units, which can usually be done as a
post-processing step.
"""
struct BandEnergyBerryVelocity{N,THV<:BandEnergyVelocity{N},TA<:ManyFourierSeries{N}} <: AbstractFourierSeries{N}
    HV::THV
    A::TA
end

function BandEnergyBerryVelocity(H::FourierSeries{N}, A::ManyFourierSeries{N,<:NTuple{N}}) where N
    @assert period(H) == period(A)
    BandEnergyBerryVelocity(BandEnergyVelocity(H), A)
end
period(f::BandEnergyBerryVelocity) = period(f.HV)
contract(f::BandEnergyBerryVelocity, x::Number) = BandEnergyBerryVelocity(contract(f.HV, x), contract(f.A, x))
function value(f::BandEnergyBerryVelocity{0})
    H, vs... = value(f.HV)
    As = value(f.A)
    (H, ntuple(n -> vs[n] - im*(H*As[n] - As[n]*H), Val{length(As)}())...)
end
Base.eltype(::Type{BandEnergyBerryVelocity{N,THV,TA}}) where {N,THV,TA} = eltype(THV)
