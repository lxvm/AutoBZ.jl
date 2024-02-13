check_degen(r_degen, nkpt) = @assert Int(sum(m -> 1//m, r_degen)) == nkpt

function parse_hamiltonian(file::IO, ::Type{F}) where {F<:AbstractFloat}
    date_time = readline(file)

    num_wann = parse(Int, readline(file))
    nrpts = parse(Int, readline(file))

    entries_per_line = 15
    degen = Vector{Int}(undef, nrpts)
    for j in 1:ceil(Int, nrpts/entries_per_line)
        col = split(readline(file)) # degeneracy of Wigner-Seitz grid points
        for i in eachindex(col)
            i > entries_per_line && error("parsing found more entries than expected")
            degen[i+(j-1)*entries_per_line] = parse(Int, col[i])
        end
    end

    C = Vector{SMatrix{num_wann,num_wann,Complex{F},num_wann^2}}(undef, nrpts)
    irvec = Vector{SVector{3,Int}}(undef, nrpts)
    c = Matrix{Complex{F}}(undef, num_wann, num_wann)
    for k in 1:nrpts
        for j in 1:num_wann^2
            col = split(readline(file))
            irvec[k] = SVector{3,Int}(parse(Int, col[1]), parse(Int, col[2]), parse(Int, col[3]))
            m = parse(Int, col[4])
            n = parse(Int, col[5])
            re = parse(F, col[6])
            im = parse(F, col[7])
            c[m,n] = complex(re, im)
        end
        C[k] = SMatrix{num_wann,num_wann,Complex{F},num_wann^2}(c)
    end
    return (; date_time, num_wann, nrpts, degen, irvec, H=C)
end

"""
    load_interp(::Type{<:HamiltonianInterp}, seed;
        gauge=Wannier(), soc=nothing,
        compact=:N, precision=Float64, droptol=eps(precision))

Load Hamiltonian coefficients from Wannier90 output `"seed_hr.dat"` into an
[`AbstractHamiltonianInterp`](@ref) that interpolates `h` with unit period. The
`gauge` keyword can be [`Wannier`](@ref) or [`Hamiltonian`](@ref) to set whether
the coefficients are interpolated in the orbital basis, i.e. as is, or rotated
into the band basis, i.e. the Hamiltonian eigenbasis. Additionally, the `soc`
keyword can specify a `SMatrix` (from StaticArrays.jl) of twice the size of the
coefficients in the `seed` file that is added to the Hamiltonian in orbital
basis. For more details see [`SOCMatrix`](@ref).

There are also several keywords to control the memory usage of the array:
`compact` which may indicate the coefficient matrices are Hermitian, `precision`
which sets the floating-point precision of the array, and `droptol` which skips
coefficients under the given relative tolerance. Possible values of `compact` are:
- `:N`: do not store the coefficients in compact form
- `:L`: store the lower triangle of the coefficients
- `:U`: store the upper triangle of the coefficients
- `:S`: store the lower triangle of the symmetrized coefficients, `(c+c')/2`
"""
function load_interp(::Type{<:HamiltonianInterp}, seed; precision=Float64, gauge=GaugeDefault(HamiltonianInterp), compact=:N, soc=nothing, droptol=eps(precision))
    (; nkpt) = parse_wout(seed * ".wout", precision)
    (; num_wann, degen, irvec, H) = parse_hamiltonian(seed * "_hr.dat", precision)
    check_degen(degen, nkpt)
    (C, origin), = load_coefficients(Val{compact}(), droptol, num_wann, irvec, degen, H)
    f = FourierSeries(C; period=freq2rad(one(precision)), offset=Tuple(-origin))
    if soc === nothing
        return HamiltonianInterp(Freq2RadSeries(f); gauge)
    else
        return SOCHamiltonianInterp(Freq2RadSeries(WrapperFourierSeries(wrap_soc, f)), soc; gauge)
    end
end

load_coeff_type(T, ::Val{:N}, n) = SMatrix{n,n,T,n^2}
load_coeff_type(T, ::Val, n) = SHermitianCompact{n,T,StaticArrays.triangularnumber(n)}

load_coeff(T, ::Val{:N}, c) = c
load_coeff(T, ::Val{:L}, c) = T(c)
load_coeff(T, ::Val{:U}, c) = T(c')
load_coeff(T, ::Val{:S}, c) = T(0.5*(c+c'))

function load_coefficients(compact, droptol, num_wann, irvec, degen, cs...)
    return load_coefficients(compact, droptol, num_wann, irvec, degen, cs)
end
function load_coefficients(compact, reltol, num_wann, irvec, degen, cs::NTuple{N}) where N
    T = load_coeff_type(eltype(eltype(eltype(cs))), compact, num_wann)
    bounds = ntuple(n -> extrema(idx -> idx[n], irvec), Val(3))
    lbounds = map(first, bounds)
    ubounds = map(last,  bounds)
    nmodes = map((lb, ub) -> ub-lb+1, lbounds, ubounds)
    Cs = ntuple(_ -> zeros(T, nmodes), Val(N))
    origins = map(C -> CartesianIndex(ntuple(n -> firstindex(C,n)-lbounds[n], Val(3))), Cs)
    for (i,idx) in enumerate(irvec)
        _idx_ = CartesianIndex(idx...)
        for (c,C,o) in zip(cs, Cs, origins)
            C[_idx_+o] = load_coeff(T, compact, c[i])/degen[i]
        end
    end
    return map((C, origin) -> droptol(C, origin, reltol), Cs, origins)
end

infnorm(x::Number) = abs(x)
infnorm(x::AbstractArray) = maximum(infnorm, x)

# based on FastChebInterp.jl and helpful for truncating zero-padded coefficients
function droptol(C::AbstractArray, origin::CartesianIndex, reltol)
    abstol = infnorm(C) * reltol
    norm_geq_tol = >=(abstol)∘infnorm
    # https://juliaarrays.github.io/OffsetArrays.jl/stable/internals/#Caveats
    # compute the new size, dropping values below tol at both ends of axes
    newlb = ntuple(ndims(C)) do dim
        n = firstindex(C, dim)
        while n < lastindex(C, dim)
            r = let n=n; ntuple(i -> axes(C,i)[i == dim ? (n:n) : (begin:end)], ndims(C)); end
            any(norm_geq_tol, @view C[CartesianIndices(r)]) && break
            n += 1
        end
        n
    end
    newub = ntuple(ndims(C)) do dim
        n = lastindex(C, dim)
        while n > firstindex(C, dim)
            r = let n=n; ntuple(i -> axes(C,i)[i == dim ? (n:n) : (begin:end)], ndims(C)); end
            any(norm_geq_tol, @view C[CartesianIndices(r)]) && break
            n -= 1
        end
        n
    end
    newC = C[CartesianIndices(map(:, newlb, newub))]
    neworigin = origin + CartesianIndex(ntuple(n -> firstindex(newC,n)-newlb[n], ndims(C)))
    return (newC, neworigin)
end

function parse_position_operator(file::IO, ::Type{F}, rot=I) where {F<:AbstractFloat}
    date_time = readline(file)

    num_wann = parse(Int, readline(file))
    nrpts = parse(Int, readline(file))

    T = SMatrix{num_wann,num_wann,Complex{F},num_wann^2}
    A1 = Vector{T}(undef, nrpts)
    A2 = Vector{T}(undef, nrpts)
    A3 = Vector{T}(undef, nrpts)
    irvec = Vector{SVector{3,Int}}(undef, nrpts)
    a1 = Matrix{Complex{F}}(undef, num_wann, num_wann)
    a2 = Matrix{Complex{F}}(undef, num_wann, num_wann)
    a3 = Matrix{Complex{F}}(undef, num_wann, num_wann)
    for k in 1:nrpts
        for j in 1:num_wann^2
            col = split(readline(file))
            irvec[k] = SVector{3,Int}(parse(Int, col[1]), parse(Int, col[2]), parse(Int, col[3]))
            m = parse(Int, col[4])
            n = parse(Int, col[5])
            re_x = parse(F, col[6])
            im_x = parse(F, col[7])
            x = complex(re_x, im_x)
            re_y = parse(F, col[8])
            im_y = parse(F, col[9])
            y = complex(re_y, im_y)
            re_z = parse(F, col[10])
            im_z = parse(F, col[11])
            z = complex(re_z, im_z)
            a1[m,n], a2[m,n], a3[m,n] = rot * SVector{3,Complex{F}}(x, y, z)
        end
        A1[k] = T(a1)
        A2[k] = T(a2)
        A3[k] = T(a3)
    end
    return (; date_time, num_wann, nrpts, irvec, As=(A1, A2, A3))
end

pick_rot(::Cartesian, A, invA) = (I, invA)
pick_rot(::Lattice, A, invA) = (invA, A)

"""
    load_interp(::Type{<:BerryConnectionInterp}, seed;
        coord=Lattice(), soc=nothing,
        precision=Float64, compact=:N, droptol=eps(precision))

Load position operator coefficients from Wannier90 output `"seed_r.dat"` into a
[`BerryConnectionInterp`](@ref) that interpolates `(A1, A2, A3)`. Specify
`coord` as [`Lattice`](@ref) or [`Cartesian`](@ref) to have the position
operator interpolated in those coordinates.
"""
function load_interp(::Type{<:BerryConnectionInterp}, seed; precision=Float64, coord=CoordDefault(BerryConnectionInterp), compact=:N, soc=nothing, droptol=eps(precision))
    (; A, nkpt) = parse_wout(seed * ".wout", precision)
    invA = inv(A) # compute inv(A) for map from Cartesian to lattice coordinates
    rot, irot = pick_rot(coord, A, invA)
    (; degen) = parse_hamiltonian(seed * "_hr.dat", precision)
    check_degen(degen, nkpt)
    (; num_wann, irvec, As) = parse_position_operator(seed * "_r.dat", precision, rot)
    (A1,o1), (A2,o2), (A3,o3) = load_coefficients(Val{compact}(), droptol, num_wann, irvec, degen, As)
    F1 = FourierSeries(A1; period=one(precision), offset=Tuple(-o1))
    F2 = FourierSeries(A2; period=one(precision), offset=Tuple(-o2))
    F3 = FourierSeries(A3; period=one(precision), offset=Tuple(-o3))
    Fs = (F1, F2, F3)
    Ms = soc === nothing ? Fs : map(f -> WrapperFourierSeries(wrap_soc, f), Fs)
    BerryConnectionInterp{coord}(ManyFourierSeries(Ms...; period=one(precision)), irot; coord)
end

"""
    load_interp(::Type{<:GradientVelocityInterp}, seed, A;
        gauge=Wannier(), vcomp=Whole(), coord=Lattice(), soc=nothing,
        precision=Float64, compact=:N, droptol=eps(precision))

Load coefficients for a Hamiltonian and its derivatives from Wannier90 output
`"seed_hr.dat"` into a [`GradientVelocityInterp`](@ref) that interpolates `(h, v)`.
Specify `vcomp` as [`Whole`](@ref), [`Intra`](@ref), or [`Inter`](@ref) to use
certain transitions. Note these velocities are not gauge-covariant.
"""
function load_interp(::Type{<:GradientVelocityInterp}, seed, A;
    precision=Float64, compact=:N, soc=nothing, droptol=eps(precision),
    gauge=GaugeDefault(GradientVelocityInterp),
    coord=CoordDefault(GradientVelocityInterp),
    vcomp=VcompDefault(GradientVelocityInterp))
    # for h require the default gauge
    h = load_interp(HamiltonianInterp, seed; precision, compact, soc)
    return GradientVelocityInterp(h, A; coord, vcomp, gauge)
end

"""
    load_interp(::Type{<:CovariantVelocityInterp}, seed, A;
        gauge=Wannier(), vcomp=whole(), coord=Lattice(), soc=nothing,
        precision=Float64, compact=:N, droptol=eps(precision))

Load coefficients for a Hamiltonian and its derivatives from Wannier90 output
`"seed_hr.dat"` and `"seed_r.dat"` into a [`CovariantVelocityInterp`](@ref) that
interpolates `(h, v)`. Specify `vcomp` as [`Whole`](@ref), [`Intra`](@ref), or
[`Inter`](@ref) to use certain transitions. These velocities are gauge-covariant.
"""
function load_interp(::Type{<:CovariantVelocityInterp}, seed, A;
    precision=Float64, compact=:N, soc=nothing, droptol=eps(precision),
    gauge=GaugeDefault(CovariantVelocityInterp),
    coord=CoordDefault(CovariantVelocityInterp),
    vcomp=VcompDefault(CovariantVelocityInterp))
    # for hv require the default gauge and vcomp
    hv = load_interp(GradientVelocityInterp, seed, A; precision, coord, compact, soc)
    a = load_interp(BerryConnectionInterp, seed; precision, coord, compact, soc)
    return CovariantVelocityInterp(hv, a; coord, vcomp, gauge)
end

"""
    load_interp(::Type{<:MassVelocityInterp}, seed, A;
        gauge=Wannier(), vcomp=Whole(), coord=Lattice(), soc=nothing,
        precision=Float64, compact=:N, droptol=eps(precision))

Load coefficients for a Hamiltonian and its derivatives from Wannier90 output
`"seed_hr.dat"` into a [`MassVelocityInterp`](@ref) that interpolates `(h, v, μ)`.
Specify `vcomp` as [`Whole`](@ref), [`Intra`](@ref), or [`Inter`](@ref) to use
certain transitions. Note these operators are not gauge-covariant.
"""
function load_interp(::Type{<:MassVelocityInterp}, seed, A;
    precision=Float64, compact=:N, soc=nothing, droptol=eps(precision),
    gauge=GaugeDefault(MassVelocityInterp),
    coord=CoordDefault(MassVelocityInterp),
    vcomp=VcompDefault(MassVelocityInterp))
    # for h require the default gauge
    h = load_interp(HamiltonianInterp, seed; precision, compact, soc, droptol)
    return MassVelocityInterp(h, A; coord, vcomp, gauge)
end

function _split!(c, s, args...; kws...)
    empty!(c)
    append!(c, eachsplit(s, args...; kws...))
end

function parse_wout(file::IO, ::Type{F}) where {F<:AbstractFloat}

    cols = SubString{String}[]
    c = zero(MMatrix{3,3,F,9})
    A = SMatrix(c)
    vol = zero(F)
    B = SMatrix(c)

    species = String[]
    site = Int[]
    frac_lat = SVector{3,F}[]
    cart_lat = SVector{3,F}[]

    nk1 = nk2 = nk3 = nkpt = 0

    # skip header
    while !eof(file)
        line = readline(file)
        if occursin("Lattice Vectors", line)
            occursin("Ang", line) || @warn "Length unit other than Angstrom detected"
            _split!(cols, line)
            for i in 1:3
                line = readline(file)
                _split!(cols, line)
                @assert popfirst!(cols) == "a_$i"
                @. c[:,i] = parse(F, cols)
            end
            A = SMatrix(c)
            continue
        end

        if occursin("Unit Cell Volume:", line)
            _split!(cols, line)
            vol = parse(F, cols[4])
            continue
        end

        if occursin("Reciprocal-Space Vectors", line)
            for i in 1:3
                line = readline(file)
                _split!(cols, line)
                @assert popfirst!(cols) == "b_$i"
                @. c[:,i] = parse(F, cols)
            end
            B = SMatrix(c)
            continue
        end

        if occursin("Site", line)
            readline(file)
            while true
                line = readline(file)
                _split!(cols, line)
                length(cols) == 11 || break
                push!(species, cols[2])
                push!(site, parse(Int, cols[3]))
                push!(frac_lat, parse.(F, cols[4:6]))
                push!(cart_lat, parse.(F, cols[8:10]))
            end
            continue
        end

        if occursin("Grid size", line)
            _split!(cols, line)
            nk1 = parse(Int, cols[4])
            nk2 = parse(Int, cols[6])
            nk3 = parse(Int, cols[8])
            nkpt = parse(Int, cols[12])
        end
    end
    return (; A, B, vol, species, site, frac_lat, cart_lat, nkpt, dims=(nk1,nk2,nk3))
end

function parse_sym(file::IO, ::Type{F}) where {F<:AbstractFloat}
    nsymmetry = parse(Int, readline(file))
    readline(file)
    point_sym = Vector{SMatrix{3,3,F,9}}(undef, nsymmetry)
    translate = Vector{SVector{3,F}}(undef, nsymmetry)
    S = Matrix{F}(undef, (3,4))
    for i in 1:nsymmetry
        for j in 1:4
            col = split(readline(file))
            S[:,j] .= parse.(F, col)
        end
        point_sym[i] = S[:,1:3]
        translate[i] = S[:,4]
        readline(file)
    end
    return (; nsymmetry, point_sym, translate)
end

load_interp(seedname::String; kwargs...) =
    load_interp(HamiltonianInterp, seedname; kwargs...)

"""
    load_autobz(::AbstractBZ, seedname; kws...)

Automatically load a BZ using data from a "seedname.wout" file with the `load_bz` interface
from AutoBZCore.
"""
function load_autobz(bz::AbstractBZ, seedname::String; precision=Float64, atol=1e-5)
    (; A, B) = parse_wout(seedname * ".wout", precision)
    return load_bz(bz, A, B; atol)
end
function load_autobz(bz::IBZ, seedname::String; precision=Float64, kws...)
    (; A, B, species, frac_lat) = parse_wout(seedname * ".wout", precision)
    return load_bz(convert(AbstractBZ{3}, bz), A, B, species, reduce(hcat, frac_lat); kws..., coordinates="lattice")
end

struct CompactDisplay{T}
    obj::T
end
Base.show(io::IO, a::CompactDisplay) = show(io, a.obj)

"""
    load_wannier90_data(seedname::String; bz::AbstractBZ=FBZ(), interp::AbstractWannierInterp=HamiltonianInterp, kwargs...)

Return a tuple `(interp, bz)` containing the requested Wannier interpolant,
`interp` and the Brillouin zone `bz` to integrate over. The `seedname` should
point to Wannier90 data to read in. Additional keywords are passed to the
interpolant constructor, [`load_interp`](@ref), while [`load_autobz`](@ref) can be
referenced for Brillouin zone details. For a list of possible keywords, see
`subtypes(AbstractBZ)` and `using TypeTree; tt(AbstractWannierInterp)`.
"""
function load_wannier90_data(seedname::String; precision=Float64, load_interp=load_interp, load_autobz=load_autobz, bz=FBZ(), interp=HamiltonianInterp, kwargs...)
    bz = load_autobz(bz, seedname; precision)
    wi = if interp <: AbstractVelocityInterp
        load_interp(interp, seedname, bz.A; precision, kwargs...)
    else
        load_interp(interp, seedname; precision, kwargs...)
    end

    if bz.syms !== nothing
        k = rand(SVector{checksquare(bz.B),eltype(bz.B)})
        ref = wi(k)
        err, idx = findmax(bz.syms) do S
            val = wi(S*k)
            return calc_interp_error(wi, val, ref)
        end
        sym = CompactDisplay(bz.syms[idx])
        kpt = CompactDisplay(k)
        msg = "Symmetry test of the interpolant's Hamiltonian at a random k-point"
        @logmsg (err > sqrt(eps(one(err)))*oneunit(err) ? Warn : Info) msg seedname kpt sym err
    end
    return (wi, bz)
end

for name in (:parse_hamiltonian, :parse_position_operator, :parse_wout, :parse_sym)
    @eval $name(filename::String, args...) = open(io -> $name(io, args...), filename)
end


# the only error we can evaluate is that of the Hamiltonian eigenvalues, since we do not
# have the orbital representation of the symmetry operators available.
# this is because H(k) and H(pg*k) are identical up to a gauge transformation
# The velocities are also tough to compare because the point-group operator should be applied
function calc_interp_error_(g::AbstractGauge, val, err)
    wval = g isa Hamiltonian ? val : to_gauge(Hamiltonian(), val)
    werr = g isa Hamiltonian ? err : to_gauge(Hamiltonian(), err)
    return norm(wval.values - werr.values)
end
function calc_interp_error(h::AbstractHamiltonianInterp, val, err)
    return calc_interp_error_(gauge(h), val, err)
end
function calc_interp_error(hv::AbstractVelocityInterp, val, err)
    return calc_interp_error_(gauge(hv), val[1], err[1])
end
calc_interp_error(::BerryConnectionInterp, val, err) = NaN

function load_wannierio_interp end
