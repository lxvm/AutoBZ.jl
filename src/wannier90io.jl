check_degen(r_degen, nkpt) = @assert Int(sum(m -> 1//m, r_degen)) == nkpt

"""
    parse_hamiltonian(filename)

Parse an ab-initio Hamiltonian output from Wannier90 into `filename`, extracting
the fields `(date_time, num_wann, nrpts, degen, irvec, C)`
"""
function parse_hamiltonian(file::IO)
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

    C = Vector{SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}}(undef, nrpts)
    irvec = Vector{SVector{3,Int}}(undef, nrpts)
    c = Matrix{ComplexF64}(undef, num_wann, num_wann)
    for k in 1:nrpts
        for j in 1:num_wann^2
            col = split(readline(file))
            irvec[k] = SVector{3,Int}(parse(Int, col[1]), parse(Int, col[2]), parse(Int, col[3]))
            m = parse(Int, col[4])
            n = parse(Int, col[5])
            re = parse(Float64, col[6])
            im = parse(Float64, col[7])
            c[m,n] = complex(re, im)
        end
        C[k] = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}(c)
    end
    return date_time, num_wann, nrpts, degen, irvec, C
end

"""
    load_hamiltonian(seed; period=1.0, compact=:N)

Load an ab-initio Hamiltonian output from Wannier90 into `filename` as an
evaluatable `FourierSeries` whose periodicity can be set by the keyword argument
`period` which defaults to setting the period along each dimension to `1.0`. To
define different periods for different dimensions, pass an `SVector` as the
`period`. To store Hermitian Fourier coefficients in compact form, use the
keyword `compact` to specify:
- `:N`: do not store the coefficients in compact form
- `:L`: store the lower triangle of the coefficients
- `:U`: store the upper triangle of the coefficients
- `:S`: store the lower triangle of the symmetrized coefficients, `(c+c')/2`
"""
function load_interp(::Type{<:HamiltonianInterp}, seed; gauge=GaugeDefault(HamiltonianInterp), period=1.0, compact=:N, soc=nothing)
    A, B, species, site, frac_lat, cart_lat, proj, kpts = parse_wout(seed * ".wout")
    date_time, num_wann, nrpts, degen, irvec, C_ = parse_hamiltonian(seed * "_hr.dat")
    check_degen(degen, kpts.nkpt)
    C = load_coefficients(Val{compact}(), num_wann, irvec, degen, C_)[1]
    offset = map(s -> -div(s,2)-1, size(C))
    f = FourierSeries(C; period=period, offset=offset)
    return soc === nothing ? HamiltonianInterp(f, gauge=gauge) : SOCHamiltonianInterp(f, soc, gauge=gauge)
end

load_coeff_type(::Val{:N}, n) = SMatrix{n,n,ComplexF64,n^2}
load_coeff_type(::Val, n) = SHermitianCompact{n,ComplexF64,StaticArrays.triangularnumber(n)}

load_coeff(T, ::Val{:N}, c) = c
load_coeff(T, ::Val{:L}, c) = T(c)
load_coeff(T, ::Val{:U}, c) = T(c')
load_coeff(T, ::Val{:S}, c) = T(0.5*(c+c'))

function load_coefficients(compact, num_wann, irvec, degen, cs...)
    return load_coefficients(compact, num_wann, irvec, degen, cs)
end
function load_coefficients(compact, num_wann, irvec, degen, cs::NTuple{N}) where N
    T = load_coeff_type(compact, num_wann)
    nmodes = zeros(Int, 3)
    for idx in irvec
        @inbounds for i in 1:3
            if (n = abs(idx[i])) > nmodes[i]
                nmodes[i] = n
            end
        end
    end
    Cs = ntuple(_ -> zeros(T, (2nmodes .+ 1)...), Val(N))
    for (i,idx) in enumerate(irvec)
        idx_ = CartesianIndex((idx .+ nmodes .+ 1)...)
        for (j,c) in enumerate(cs)
            Cs[j][idx_] = load_coeff(T, compact, c[i])/degen[i]
        end
    end
    return Cs
end


"""
    parse_position_operator(filename, rot=I)

Parse a position operator output from Wannier90 into `filename`, extracting the
fields `(date_time, num_wann, nrpts, irvec, A1, A2, A3)`. By default, `A1, A2,
A3` are in the Cartesian basis (i.e. `X, Y, Z` because the Wannier90
`seedname_r.dat` file is), however a rotation matrix `rot` can be applied to
change the basis of the input to other coordinates.
"""
function parse_position_operator(file::IO, rot=I)
    date_time = readline(file)

    num_wann = parse(Int, readline(file))
    nrpts = parse(Int, readline(file))

    T = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}
    A1 = Vector{T}(undef, nrpts)
    A2 = Vector{T}(undef, nrpts)
    A3 = Vector{T}(undef, nrpts)
    irvec = Vector{SVector{3,Int}}(undef, nrpts)
    a1 = Matrix{ComplexF64}(undef, num_wann, num_wann)
    a2 = Matrix{ComplexF64}(undef, num_wann, num_wann)
    a3 = Matrix{ComplexF64}(undef, num_wann, num_wann)
    for k in 1:nrpts
        for j in 1:num_wann^2
            col = split(readline(file))
            irvec[k] = SVector{3,Int}(parse(Int, col[1]), parse(Int, col[2]), parse(Int, col[3]))
            m = parse(Int, col[4])
            n = parse(Int, col[5])
            re_x = parse(Float64, col[6])
            im_x = parse(Float64, col[7])
            x = complex(re_x, im_x)
            re_y = parse(Float64, col[8])
            im_y = parse(Float64, col[9])
            y = complex(re_y, im_y)
            re_z = parse(Float64, col[10])
            im_z = parse(Float64, col[11])
            z = complex(re_z, im_z)
            a1[m,n], a2[m,n], a3[m,n] = rot * SVector{3,ComplexF64}(x, y, z)
        end
        A1[k] = T(a1)
        A2[k] = T(a2)
        A3[k] = T(a3)
    end
    return date_time, num_wann, nrpts, irvec, A1, A2, A3
end

pick_rot(::Cartesian, A, invA) = (I, invA)
pick_rot(::Lattice, A, invA) = (invA, A)

"""
    load_position_operator(seed; period=1.0, compact=nothing)

Load a position operator Hamiltonian output from Wannier90 into `filename` as an
evaluatable `ManyFourierSeries` with separate x, y, and z components whose
periodicity can be set by the keyword argument `period` which defaults to
setting the period along each dimension to `1.0`. To define different periods
for different dimensions, pass an `SVector` as the `period`. To store Hermitian
Fourier coefficients in compact form, use the keyword `compact` to specify:
- `:N`: do not store the coefficients in compact form
- `:L`: store the lower triangle of the coefficients
- `:U`: store the upper triangle of the coefficients
- `:S`: store the lower triangle of the symmetrized coefficients, `(c+c')/2`
Note that in some cases the coefficients are not Hermitian even though the
values of the series are.
"""
function load_interp(::Type{<:BerryConnectionInterp}, seed; coord=CoordDefault(BerryConnectionInterp), period=1.0, compact=:N, soc=nothing)
    A, B, species, site, frac_lat, cart_lat, proj, kpts = parse_wout(seed * ".wout")
    invA = inv(A) # compute inv(A) for map from Cartesian to lattice coordinates
    rot, irot = pick_rot(coord, A, invA)
    date_time, num_wann, nrpts, degen, irvec, C_ = parse_hamiltonian(seed * "_hr.dat")
    check_degen(degen, kpts.nkpt)
    date_time, num_wann, nrpts, irvec, A1_, A2_, A3_ = parse_position_operator(seed * "_r.dat", rot)
    A1, A2, A3 = load_coefficients(Val{compact}(), num_wann, irvec, degen, A1_, A2_, A3_)
    F1_ = FourierSeries(A1; period=period, offset=map(s -> -div(s,2)-1, size(A1)))
    F1  = soc === nothing ? F1_ : WrapperSeries(F1_, wrap_soc)
    F2_ = FourierSeries(A2; period=period, offset=map(s -> -div(s,2)-1, size(A2)))
    F2  = soc === nothing ? F2_ : WrapperSeries(F2_, wrap_soc)
    F3_ = FourierSeries(A3; period=period, offset=map(s -> -div(s,2)-1, size(A3)))
    F3  = soc === nothing ? F3_ : WrapperSeries(F3_, wrap_soc)
    BerryConnectionInterp{coord}(ManyFourierSeries(F1, F2, F3; period=period), irot; coord=coord)
end

"""
    load_gradient_hamiltonian_velocities(seed; period=1.0, compact=:N,
    gauge=Wannier(), vcomp=Whole(), coord=Lattice())

Load the Hamiltonian and band velocities, which may later be passed to one of
the integrand constructors. When called with one filename, that file is parsed
as a Wannier 90 Hamiltonian and the resulting band velocities are just the
gradient of the Hamiltonian. The return type is [`HamiltonianVelocity3D`](@ref). When
called with two filenames, the second is parsed as a position operator from
Wannier 90 and adds a contribution to band velocities from the Berry connection.
The return type is [`CovariantHamiltonianVelocity3D`](@ref).The keywords `period` and
`compact` set the reciprocal unit cell length and whether the coefficients of
the Fourier series should be compressed as Hermitian matrices. Typically the
coefficients cannot be compressed despite the values of the series being
Hermitian. The keyword `gauge` can take values of `:Wannier` and `:Hamiltonian`
and the keyword `vcomp` can take values `:whole`, `:inter` and `:intra`. See
[`AutoBZ.Jobs.to_gauge`](@ref) and [`AutoBZ.Jobs.band_velocities`](@ref) for
details.
"""
function load_interp(::Type{<:GradientVelocityInterp}, seed, A=parse_wout(seed * ".wout")[1]; period=1.0, compact=:N, soc=nothing,
    gauge=GaugeDefault(GradientVelocityInterp),
    coord=CoordDefault(GradientVelocityInterp),
    vcomp=VcompDefault(GradientVelocityInterp))
    # for h require the default gauge
    h = load_interp(HamiltonianInterp, seed; period=period, compact=compact, soc=soc)
    return GradientVelocityInterp(h, A; coord=coord, vcomp=vcomp, gauge=gauge)
end

"""
    load_covariant_hamiltonian_velocities(seed; period=1.0, compact=:N,
    gauge=Wannier(), vcomp=whole(), coord=Lattice())
"""
function load_interp(::Type{<:CovariantVelocityInterp}, seed, A=parse_wout(seed * ".wout")[1]; period=1.0, compact=:N, soc=nothing,
    gauge=GaugeDefault(CovariantVelocityInterp),
    coord=CoordDefault(CovariantVelocityInterp),
    vcomp=VcompDefault(CovariantVelocityInterp))
    # for hv require the default gauge and vcomp
    hv = load_interp(GradientVelocityInterp, seed, A; coord=coord, period=period, compact=compact, soc=soc)
    a = load_interp(BerryConnectionInterp, seed; coord=coord, period=period, compact=compact, soc=soc)
    return CovariantVelocityInterp(hv, a; coord=coord, vcomp=vcomp, gauge=gauge)
end

function load_interp(::Type{<:MassVelocityInterp}, seed, A=parse_wout(seed * ".wout")[1]; period=1.0, compact=:N, soc=nothing,
    gauge=GaugeDefault(MassVelocityInterp),
    coord=CoordDefault(MassVelocityInterp),
    vcomp=VcompDefault(MassVelocityInterp))
    # for h require the default gauge
    h = load_interp(HamiltonianInterp, seed; period=period, compact=compact, soc=soc)
    return MassVelocityInterp(h, A; coord=coord, vcomp=vcomp, gauge=gauge)
end

"""
    parse_wout(filename; iprint=1)

returns the lattice vectors `a` and reciprocal lattice vectors `b`
"""
function parse_wout(file::IO)

    # header
    while (l = strip(readline(file))) != "SYSTEM"
        continue
    end

    # system
    readline(file)
    readline(file)
    readline(file)
    ## lattice vectors
    c = Matrix{Float64}(undef, 3, 3)
    for i in 1:3
        col = split(readline(file))
        popfirst!(col)
        @. c[:,i] = parse(Float64, col)
    end
    A = SMatrix{3,3,Float64,9}(c)

    readline(file)
    readline(file)
    readline(file)
    readline(file)
    ## reciprocal lattice vectors
    for i in 1:3
        col = split(readline(file))
        popfirst!(col)
        @. c[:,i] = parse(Float64, col)
    end
    B = SMatrix{3,3,Float64,9}(c)


    readline(file)
    readline(file)
    readline(file) # site fractional coordinate cartesian coordinate (unit)
    readline(file)
    # lattice
    species = String[]
    site = Int[]
    frac_lat_ = SVector{3,Float64}[]
    cart_lat_ = SVector{3,Float64}[]
    while true
        col = split(readline(file))
        length(col) == 11 || break
        push!(species, col[2])
        push!(site, parse(Int, col[3]))
        push!(frac_lat_, parse.(Float64, col[4:6]))
        push!(cart_lat_, parse.(Float64, col[8:10]))
    end
    frac_lat = Matrix(reshape(reinterpret(Float64, frac_lat_), 3, :))
    cart_lat = Matrix(reshape(reinterpret(Float64, cart_lat_), 3, :))

    readline(file)
    readline(file) # projections
    readline(file)
    readline(file)
    readline(file)
    readline(file) # Frac. coord, l, mr, r, z-axis, x-axis, Z/a
    readline(file)

    pfrac = SVector{3,Float64}[]
    l = Int[]
    mr = Int[]
    r = Int[]
    zax = SVector{3,Float64}[]
    xax = SVector{3,Float64}[]
    za = Float64[]
    while true
        col = split(readline(file))
        length(col) == 15 || break
        push!(pfrac, SVector{3,Float64}(parse.(Float64, col[2:4])))
        push!(l, parse(Int, col[5]))
        push!(mr, parse(Int, col[6]))
        push!(r, parse(Int, col[7]))
        push!(zax, SVector{3,Float64}(parse.(Float64, col[8:10])))
        push!(xax, SVector{3,Float64}(parse.(Float64, col[11:13])))
        push!(za, parse(Float64, col[14]))
    end
    proj = (; frac=pfrac, l=l, mr=mr, r=r, zax=zax, xax=xax, za=za)

    # k-point grid
    readline(file)
    readline(file)
    readline(file) # k-point grid
    readline(file)
    readline(file)

    kpt_header = split(readline(file))
    sizes = (parse(Int, kpt_header[4]), parse(Int, kpt_header[6]), parse(Int, kpt_header[8]))
    nkpt = parse(Int, kpt_header[12])
    readline(file)
    readline(file)
    readline(file)  # k-point, fractional coordinate, Cartesian coordinate (Ang^-1)
    readline(file)
    ikvec = Vector{Int}(undef, nkpt)
    kfrac = Vector{SVector{3,Float64}}(undef, nkpt)
    kcart = Vector{SVector{3,Float64}}(undef, nkpt)
    for i in 1:nkpt
        col = split(readline(file))
        ikvec[i] = parse(Int, col[2])
        kfrac[i] = parse.(Float64, col[3:5])
        kcart[i] = parse.(Float64, col[7:9])
    end
    kpts = (; sizes=sizes, nkpt=nkpt, ikvec=ikvec, cart=kcart, frac=kfrac)

    # main
    # wannierise
    # disentangle
    # plotting
    # k-mesh
    # etc...

    return A, B, species, site, frac_lat, cart_lat, proj, kpts
end

function parse_sym(file::IO)
    nsymmetry = parse(Int, readline(file))
    readline(file)
    point_sym = Vector{SMatrix{3,3,Float64,9}}(undef, nsymmetry)
    translate = Vector{SVector{3,Float64}}(undef, nsymmetry)
    S = Matrix{Float64}(undef, (3,4))
    for i in 1:nsymmetry
        for j in 1:4
            col = split(readline(file))
            S[:,j] .= parse.(Float64, col)
        end
        point_sym[i] = S[:,1:3]
        translate[i] = S[:,4]
        readline(file)
    end
    return nsymmetry, point_sym, translate
end

load_interp(seedname::String; kwargs...) =
    load_interp(HamiltonianInterp, seedname; kwargs...)

"""
    load_autobz(::AbstractBZ, seedname; kws...)

Automatically load a BZ using data from a "seedname.wout" file with the `load_bz` interface
from AutoBZCore.
"""
function load_autobz(bz::AbstractBZ, seedname::String; atol=1e-5)
    A, B, = parse_wout(seedname * ".wout")
    return load_bz(bz, A, B; atol=atol)
end
function load_autobz(bz::IBZ, seedname::String; kws...)
    a, b, species, site, frac_lat, cart_lat = parse_wout(seedname * ".wout")
    return load_bz(convert(AbstractBZ{3}, bz), a, b, species, frac_lat; kws..., coordinates="lattice")
end

"""
    load_wannier90_data(seedname::String; bz::AbstractBZ=FBZ(), interp::AbstractWannierInterp=HamiltonianInterp, kwargs...)

Return a tuple `(interp, bz)` containing the requested Wannier interpolant,
`interp` and the Brillouin zone `bz` to integrate over. The `seedname` should
point to Wannier90 data to read in. Additional keywords are passed to the
interpolant constructor, [`load_interp`](@ref), while [`load_autobz`](@ref) can be
referenced for Brillouin zone details. For a list of possible keywords, see
`subtypes(AbstractBZ)` and `using TypeTree; tt(AbstractWannierInterp)`.
"""
function load_wannier90_data(seedname::String; load_interp=load_interp, load_autobz=load_autobz, bz=FBZ(), interp=HamiltonianInterp, kwargs...)
    bz = load_autobz(bz, seedname)
    wi = if interp <: AbstractVelocityInterp
        load_interp(interp, seedname, bz.A; kwargs...)
    else
        load_interp(interp, seedname; kwargs...)
    end

    if bz.syms !== nothing
        k = rand(SVector{checksquare(bz.B),eltype(bz.B)})
        ref = wi(k)
        err, idx = findmax(bz.syms) do S
            val = wi(S*k)
            return calc_interp_error(wi, val, ref)
        end
        msg = """
        Testing that the interpolant's Hamiltonian satisfies the symmetries at random k-point
            k = $(k)
        and found a maximum error $(err) for symmetry
            bz.syms[$(idx)] = $(bz.syms[idx])
        """
        err > 1e-5 ? @warn(msg) : @info(msg)
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
