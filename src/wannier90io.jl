"""
    parse_hamiltonian(filename)

Parse an ab-initio Hamiltonian output from Wannier90 into `filename`, extracting
the fields `(date_time, num_wann, nrpts, degen, irvec, C)`
"""
parse_hamiltonian(filename) = open(filename) do file
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
    date_time, num_wann, nrpts, degen, irvec, C
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
function load_hamiltonian(seed; gauge=:Wannier, period=1.0, compact=:N)
    date_time, num_wann, nrpts, degen, irvec, C_ = parse_hamiltonian(seed * "_hr.dat")
    C = load_coefficients(Val{compact}(), num_wann, irvec, C_)[1]
    f = InplaceFourierSeries(C; period=period, offset=map(s -> -div(s,2)-1, size(C)))
    Hamiltonian(f; gauge=gauge)
end

load_coefficients(compact, num_wann, irvec, cs...) = load_coefficients(compact, num_wann, irvec, cs)
@generated function load_coefficients(compact, num_wann, irvec, cs::NTuple{N}) where N
    T_full = :(SMatrix{num_wann,num_wann,ComplexF64,num_wann^2})
    T_compact = :(SHermitianCompact{num_wann,ComplexF64,StaticArrays.triangularnumber(num_wann)})
    if compact === Val{:N}
        T = T_full; expr = :(c[i])
    elseif compact === Val{:L}
        T = T_compact; expr = :($T(c[i]))
    elseif compact === Val{:U}
        T = T_compact; expr = :($T(c[i]'))
    elseif compact === Val{:S}
        T = T_compact; expr = :($T(0.5*(c[i]+c[i]')))
    end
    quote
        nmodes = zeros(Int, 3)
        for idx in irvec
            @inbounds for i in 1:3
                if (n = abs(idx[i])) > nmodes[i]
                    nmodes[i] = n
                end
            end
        end
        Cs = Base.Cartesian.@ntuple $N _ -> zeros($T, (2nmodes .+ 1)...)
        for (i,idx) in enumerate(irvec)
            idx_ = CartesianIndex((idx .+ nmodes .+ 1)...)
            for (j,c) in enumerate(cs)
                Cs[j][idx_] = $expr
            end
        end
        Cs
    end
end


"""
    parse_position_operator(filename)

Parse a position operator output from Wannier90 into `filename`, extracting the
fields `(date_time, num_wann, nrpts, irvec, X, Y, Z)`
"""
parse_position_operator(filename) = open(filename) do file
    date_time = readline(file)

    num_wann = parse(Int, readline(file))
    nrpts = parse(Int, readline(file))

    T = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}
    X = Vector{T}(undef, nrpts)
    Y = Vector{T}(undef, nrpts)
    Z = Vector{T}(undef, nrpts)
    irvec = Vector{SVector{3,Int}}(undef, nrpts)
    x = Matrix{ComplexF64}(undef, num_wann, num_wann)
    y = Matrix{ComplexF64}(undef, num_wann, num_wann)
    z = Matrix{ComplexF64}(undef, num_wann, num_wann)
    for k in 1:nrpts
        for j in 1:num_wann^2
            col = split(readline(file))
            irvec[k] = SVector{3,Int}(parse(Int, col[1]), parse(Int, col[2]), parse(Int, col[3]))
            m = parse(Int, col[4])
            n = parse(Int, col[5])
            re_x = parse(Float64, col[6])
            im_x = parse(Float64, col[7])
            x[m,n] = complex(re_x, im_x)
            re_y = parse(Float64, col[8])
            im_y = parse(Float64, col[9])
            y[m,n] = complex(re_y, im_y)
            re_z = parse(Float64, col[10])
            im_z = parse(Float64, col[11])
            z[m,n] = complex(re_z, im_z)
        end
        X[k] = T(x)
        Y[k] = T(y)
        Z[k] = T(z)
    end
    date_time, num_wann, nrpts, irvec, X, Y, Z
end


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
function load_position_operator(seed; period=1.0, compact=:N)
    date_time, num_wann, nrpts, irvec, X_, Y_, Z_ = parse_position_operator(seed * "_r.dat")
    X, Y, Z = load_coefficients(Val{compact}(), num_wann, irvec, X_, Y_, Z_)
    FX = InplaceFourierSeries(X; period=period, offset=map(s -> -div(s,2)-1, size(X)))
    FY = InplaceFourierSeries(Y; period=period, offset=map(s -> -div(s,2)-1, size(Y)))
    FZ = InplaceFourierSeries(Z; period=period, offset=map(s -> -div(s,2)-1, size(Z)))
    ManyFourierSeries(FX, FY, FZ)
end

"""
    load_gradient_hamiltonian_velocities(seed; period=1.0, compact=:N, gauge=:Wannier, vcomp=:whole)

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
function load_gradient_hamiltonian_velocities(seed; period=1.0, compact=:N, gauge=:Wannier, vcord=:lattice, vcomp=:whole)
    h = load_hamiltonian(seed; period=period, compact=compact)
    HamiltonianVelocity(h, parse_wout(seed * ".wout")[1]; vcord=vcord, vcomp=vcomp, gauge=gauge)
end

"""
    load_covariant_hamiltonian_velocities(seed; period=1.0, compact=:N, gauge=:Wannier, vcomp=:whole)
"""
function load_covariant_hamiltonian_velocities(seed; period=1.0, compact=:N, gauge=:Wannier, vcord=:lattice, vcomp=:whole)
    hv = load_gradient_hamiltonian_velocities(seed; period=period, compact=compact)
    a = load_position_operator(seed; period=period, compact=compact)
    CovariantHamiltonianVelocity(hv, a; vcord=vcord, vcomp=vcomp, gauge=gauge)
end

"""
    parse_wout(filename; iprint=1)

returns the lattice vectors `a` and reciprocal lattice vectors `b`
"""
parse_wout(filename; iprint=1) = open(filename) do file
    iprint != 1 && throw(ArgumentError("Verbosity setting iprint not implemented"))

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
    # projections
    # k-point grid
    # main
    # wannierise
    # disentangle
    # plotting
    # k-mesh
    # etc...

    return A, B, species, site, frac_lat, cart_lat
end

parse_sym(filename) = open(filename) do file
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

"""
    load_wannier90_data(seedname; gauge=:Wannier, vkind=:none, vcord=:lattice, vcomp=:whole, compact=:N, bzkind=:full)

Reads Wannier90 output files with the given `seedname` to return the Hamiltonian
(optionally with band velocities if `vkind` is specified as either `:covariant`
or `:gradient`) and the full Brillouin zone limits. The keyword `compact` is
available if to compress the Fourier series if its Fourier coefficients are
known to be Hermitian. Returns `(w, fbz)` containing the Wannier interpolant and
the full BZ limits.
"""
function load_wannier90_data(seedname::String; gauge=:Wannier, vkind=:none, vcord=:lattice, vcomp=:whole, compact=:N, bzkind=:fbz)
    # use fractional lattice coordinates for the Fourier series
    bz = load_bz(Val(bzkind), seedname)

    w = if vkind == :none
        load_hamiltonian(seedname * "_hr.dat"; compact=compact, gauge=gauge)
    elseif vkind == :covariant
        load_covariant_hamiltonian_velocities(seedname; compact=compact, gauge=gauge, vcord=vcord, vcomp=vcomp)
    elseif vkind == :gradient
        @warn "Using non-gauge-covariant velocities"
        load_gradient_hamiltonian_velocities(seedname; compact=compact, gauge=gauge, vcord=vcord, vcomp=vcomp)
    else
        throw(ArgumentError("velocity kind $vkind not recognized"))
    end

    w, bz
end