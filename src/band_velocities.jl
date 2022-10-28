export BandEnergyVelocity

"""
    BandEnergyVelocity(H::FourierSeries{3})

This constructor takes a Fourier series representing the Hamiltonian and also
evaluates the band velocities so that the return value after all the dimensions
are contracted is a tuple containing `(H, ν₁, ν₂, ν₃)`. The band velocities are
defined by dipole operators ``\\nu_{\\alpha} = -\\frac{i}{\\hbar}
\\partial_{k_{\\alpha}} H`` where ``k_{\\alpha}`` is one of three input
dimensions of ``H`` and ``\\hbar=1``. Note that differentiation changes the
units to have an additional dimension of length, so if ``H`` has units of
dimensions of energy, ``\\nu`` has dimensions of energy times length. The caller
is responsible for transforming the units of the velocity (i.e. ``\\hbar``) if
they want other units, which can usually be done as a post-processing step.
"""
BandEnergyVelocity(H::FourierSeries{3}) = BandEnergyVelocity3(H)

struct BandEnergyVelocity3{TH} <: AbstractFourierSeries{3}
    H::TH
end

Base.eltype(f::BandEnergyVelocity3) = Tuple{ntuple(_ -> eltype(f.H), Val{4}())...}
period(f::BandEnergyVelocity3) = period(f.H)

function contract(f::BandEnergyVelocity3, z::Number)
    ν₃ = FourierSeriesDerivative(f.H, SVector(0,0,1))
    BandEnergyVelocity2(contract(f.H, z), contract(ν₃, z))
end

struct BandEnergyVelocity2{TH,T3} <: AbstractFourierSeries{2}
    H::TH
    ν₃::T3
end

function contract(f::BandEnergyVelocity2, y::Number)
    ν₂ = FourierSeriesDerivative(f.H, SVector(0,1))
    BandEnergyVelocity1(contract(f.H, y), contract(ν₂, y), contract(f.ν₃, y))
end

struct BandEnergyVelocity1{TH,T2,T3} <: AbstractFourierSeries{1}
    H::TH
    ν₂::T2
    ν₃::T3
end

function contract(f::BandEnergyVelocity1, x::Number)
    ν₁ = FourierSeriesDerivative(f.H, SVector(1))
    BandEnergyVelocity0(contract(f.H, x), contract(ν₁, x), contract(f.ν₂, x), contract(f.ν₃, x))
end

struct BandEnergyVelocity0{TH,T1,T2,T3} <: AbstractFourierSeries{0}
    H::TH
    ν₁::T1
    ν₂::T2
    ν₃::T3
end
Base.eltype(f::BandEnergyVelocity0) = Tuple{map(eltype, (f.H, f.ν₁, f.ν₂, f.ν₃))...}
value(f::BandEnergyVelocity0) = (value(f.H), map(ν -> -im*I * ν, map(value, (f.ν₁, f.ν₂, f.ν₃)))...)