export BandEnergyVelocity, BandEnergyBerryVelocity

"""
    band_velocities(::Type{Val{kind}}, H, vs...) where kind

Transform the band velocities according to the following values of `kind`
- `:orbital`: return the orbital-basis velocity (i.e. null-op, stays in Wannier gauge)
- `:band`: return the band-basis velocity (transforms to Hamiltonian gauge)
- `:intraband`: return only the diagonal of the band velocity (Hamiltonian gauge)
- `:interband`: return only the off-diagonal terms of the band velocity (Hamiltonian gauge)
"""
band_velocities(kind::T, H, vs::AbstractMatrix...) where T = band_velocities(kind, H, vs)

band_velocities(::Val{:orbital}, H, vs::NTuple) = (H, vs...)
band_velocities(::Val{:band}, H, vws::NTuple) = to_hamiltonian_gauge(H, vws)

function band_velocities(::Val{:interband}, H, vws::NTuple{N,T}) where {N,T}
    ϵ, vhs... = to_hamiltonian_gauge(H, vws)
    return (ϵ, ntuple(n -> vhs[n] - Diagonal(vhs[n]), Val{N}())...)
end

function band_velocities(::Val{:intraband}, H, vws::NTuple{N}) where N
    ϵ, vhs... = to_hamiltonian_gauge(H, vws)
    return (ϵ, ntuple(n -> Diagonal(vhs[n]), Val{N}())...)
end

function to_hamiltonian_gauge(H, vws::NTuple{N}) where {N}
    ishermitian(H) || throw(ArgumentError("found non-Hermitian Hamiltonian"))
    vals, U = eigen(Hermitian(H)) # need to wrap with Hermitian for type stability
    (Diagonal(vals), ntuple(n -> U'*vws[n]*U, Val{N}())...)
end


"""
    BandEnergyVelocity{kind}(H::FourierSeries{N}, V::ManyFourierSeries{N}) where {kind,N}
    BandEnergyVelocity(H::FourierSeries{N}; kind=:orbital) where N

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
struct BandEnergyVelocity{kind,N,TH<:FourierSeries{N},TV<:ManyFourierSeries{N}} <: AbstractFourierSeries{N}
    H::TH
    V::TV
    function BandEnergyVelocity{kind}(H::TH, V::TV) where {kind,N,TH<:FourierSeries{N},TV<:ManyFourierSeries{N}}
        new{kind,N,TH,TV}(H, V)
    end
end
BandEnergyVelocity(H::FourierSeries{N}; kind=:orbital) where N = BandEnergyVelocity{Val{kind}()}(H, ManyFourierSeries((), period(H)))
period(f::BandEnergyVelocity) = period(f.H)
function contract(f::BandEnergyVelocity{kind,N}, x::Number) where {kind,N}
    v = FourierSeriesDerivative(f.H, SVector(ntuple(n -> ifelse(n == N, 1, 0), Val{N}())))
    BandEnergyVelocity{kind}(contract(f.H, x), ManyFourierSeries((contract(v, x), contract(f.V, x).fs...), pop(period(f))))
end
function contract(f::BandEnergyVelocity{kind,1}, x::Number) where kind
    v = FourierSeriesDerivative(f.H, SVector(1))
    #= we precompute the (potentially costly) velocities here rather than in
    value because routines may call value on the same data many times =#
    H, vs... = band_velocities(kind, f.H(x), v(x), map(v_ -> v_(x), f.V.fs)...)
    p = pop(period(f))
    BandEnergyVelocity{kind}(FourierSeries(H, p), ManyFourierSeries(map(v -> FourierSeries(v, p), vs), p))
end
value(f::BandEnergyVelocity{<:Any,0}) = (value(f.H), value(f.V)...)
Base.eltype(::Type{BandEnergyVelocity{Val{:orbital}(),N,TH,TV}}) where {N,M,TH,TV<:ManyFourierSeries{N,<:NTuple{M}}} = NTuple{N+M+1,eltype(TH)}
Base.eltype(::Type{BandEnergyVelocity{Val{:band}(),N,TH,TV}}) where {N,M,TH,TV<:ManyFourierSeries{N,<:NTuple{M}}} = Tuple{diagonal_type(real(eltype(TH))), ntuple(i -> eltype(TH), Val{N+M}())...}
Base.eltype(::Type{BandEnergyVelocity{Val{:interband}(),N,TH,TV}}) where {N,M,TH,TV<:ManyFourierSeries{N,<:NTuple{M}}} = Tuple{diagonal_type(real(eltype(TH))), ntuple(i -> eltype(TH), Val{N+M}())...}
Base.eltype(::Type{BandEnergyVelocity{Val{:intraband}(),N,TH,TV}}) where {N,M,TH,TV<:ManyFourierSeries{N,<:NTuple{M}}} = Tuple{diagonal_type(real(eltype(TH))), ntuple(i -> diagonal_type(eltype(TH)), Val{N+M}())...}

diagonal_type(::Type{TA}) where {TA<:AbstractMatrix} = Base.promote_op(Diagonal, TA)

"""
    BandEnergyBerryVelocity{kind}(H::BandEnergyVelocity{Val{:orbital}(),N}, A::ManyFourierSeries{N}) where {kind,N}
    BandEnergyBerryVelocity(H::BandEnergyVelocity{kind,N}, A::ManyFourierSeries{N,<:NTuple{N}}) where {kind,N}
    BandEnergyBerryVelocity(H::FourierSeries, A; kind=:orbital)

This constructor takes a `FourierSeries`, `H`, representing the Hamiltonian and
also a `ManyFourierSeries`, `A`, representing the gradient of the Berry
connection, and evaluates modified band velocities so that the return value
after all the dimensions are contracted is a tuple containing `(H, ṽ₁, ṽ₂,
..., ṽₙ)`. The modified band velocities are defined by
```math
\\tilde{\\nu}_{\\alpha} = \\frac{1}{\\hbar} \\partial_{k_{\\alpha}} H -
\\frac{i}{\\hbar} [H,(A_{\\alpha} + A_{\\alpha}^{\\dagger})/2]
```
where ``k_{\\alpha}`` is one of three input dimensions of ``H`` and
``\\hbar=1``. Effectively, this type evaluates the Hamiltonian and its gradient
modified by a commutator of the Hamiltonian with the gradient of the Berry
connection. Note that differentiation by ``k`` changes the units to have an
additional dimension of length and a factor of ``2\\pi``, so if ``H`` has
dimensions of energy, ``\\nu`` has dimensions of energy times length. The caller
is responsible for transforming the units of the velocity (i.e. ``\\hbar``) if
they want other units, which can usually be done as a post-processing step.
"""
struct BandEnergyBerryVelocity{kind,N,THV<:BandEnergyVelocity{Val{:orbital}(),N},TA<:ManyFourierSeries{N}} <: AbstractFourierSeries{N}
    HV::THV
    A::TA
    function BandEnergyBerryVelocity{kind}(HV::THV, A::TA) where {kind,N,THV<:BandEnergyVelocity{Val{:orbital}(),N},TA<:ManyFourierSeries{N}}
        new{kind,N,THV,TA}(HV, A)
    end
end

BandEnergyBerryVelocity(H::FourierSeries, A; kind=:orbital) = BandEnergyBerryVelocity(BandEnergyVelocity(H; kind=kind), A)
function BandEnergyBerryVelocity(HV::BandEnergyVelocity{kind,N}, A::ManyFourierSeries{N,<:NTuple{N}}) where {kind,N}
    @assert period(HV) == period(A)
    BandEnergyBerryVelocity{kind}(BandEnergyVelocity{Val{:orbital}()}(HV.H, HV.V), A)
end
period(f::BandEnergyBerryVelocity) = period(f.HV)
contract(f::BandEnergyBerryVelocity{kind}, x::Number) where kind = BandEnergyBerryVelocity{kind}(contract(f.HV, x), contract(f.A, x))
function contract(f::BandEnergyBerryVelocity{kind,1}, x::Number) where kind
    v = FourierSeriesDerivative(f.HV.H, SVector(1))
    As = map(herm, f.A(x)) # we take the Hermitian part of the Berry connection since Wannier 90 may not have
    Hw, vws... = f.HV(x)
    # compute the band velocities from the Wannier-basis quantities
    # with a velocity modification by the Berry connection
    H, vs... = band_velocities(kind, Hw, ntuple(n -> vws[n] - (im*I)*commutator(Hw, As[n]), Val{length(As)}())...)
    p = pop(period(f))
    BV = BandEnergyVelocity{Val{:orbital}()}(FourierSeries(H, p), ManyFourierSeries(map(v -> FourierSeries(v, p), vs), p))
    BandEnergyBerryVelocity{kind}(BV, ManyFourierSeries((), p))
end
value(f::BandEnergyBerryVelocity) = value(f.HV)
Base.eltype(::Type{BandEnergyBerryVelocity{kind,N,BandEnergyVelocity{Val{:orbital}(),N,TH,TV},TA}}) where {kind,N,TH,TV,TA} = eltype(BandEnergyVelocity{kind,N,TH,TV})

