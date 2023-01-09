export load_wannier90_data

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
    load_hamiltonian(filename; period=1.0, compact=:N)

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
function load_hamiltonian(filename; period=1.0, compact=:N)
    date_time, num_wann, nrpts, degen, irvec, C_ = parse_hamiltonian(filename)
    C = load_coefficients(Val{compact}(), num_wann, irvec, C_)[1]
    FourierSeries3D(C, to_3period(period))
end

to_3period(x::Real) = to_3period(convert(Float64, x))
to_3period(x::Float64) = (x,x,x)
to_3period(x::NTuple{3,Float64}) = x

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
    load_position_operator(filename; period=1.0, compact=nothing)

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
function load_position_operator(filename; period=1.0, compact=:N)
    date_time, num_wann, nrpts, irvec, X_, Y_, Z_ = parse_position_operator(filename)
    X, Y, Z = load_coefficients(Val{compact}(), num_wann, irvec, X_, Y_, Z_)
    periods = to_3period(period)
    FX = FourierSeries3D(X, periods)
    FY = FourierSeries3D(Y, periods)
    FZ = FourierSeries3D(Z, periods)
    FX, FY, FZ
end

"""
    load_hamiltonian_velocities(f_hamiltonian, [f_pos_op]; period=1.0, compact=:N, kind=:full)

Load the Hamiltonian and band velocities, which may later be passed to one of
the integrand constructors. When called with one filename, that file is parsed
as a Wannier 90 Hamiltonian and the resulting Band velocities are just the
gradient of the Hamiltonian. The return type is `BandEnergyVelocity3D`. When
called with two filenames, the second is parsed as a position operator from
Wannier 90 and adds a contribution to band velocities from the Berry connection.
The return type is `BandEnergyBerryVelocity3D`.The keywords `period` and
`compact` set the reciprocal unit cell length and whether the coefficients of
the Fourier series should be compressed as Hermitian matrices. Typically the
coefficients cannot be compressed despite the values of the series being
Hermitian. The keyword `kind` can take values `:full`, `:inter`, and `:intra`,
referring to whether the Hamiltonian and band velocities are kept in the Wannier
gauge, or rotated to the Hamiltonian gauge to take just the inter/intra-band
part (i.e. the off-diagonal/diagonal part of the band velocities, respectively).

Warning: a `kind` of `:inter` or `:intra` is incompatible with a non-scalar Σ
"""
function load_hamiltonian_velocities(f_hamiltonian; period=1.0, compact=:N, kind=:orbital)
    H = load_hamiltonian(f_hamiltonian; period=period, compact=compact)
    BandEnergyVelocity3D(H, kind)
end
function load_hamiltonian_velocities(f_hamiltonian, f_pos_op; period=1.0, compact=:N, kind=:orbital)
    H = load_hamiltonian(f_hamiltonian; period=period, compact=compact)
    Ax, Ay, Az = load_position_operator(f_pos_op; period=period, compact=compact)
    BandEnergyBerryVelocity3D(H, Ax, Ay, Az, kind)
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
    a = Vector{SVector{3,Float64}}(undef, 3)
    for i in 1:3
        col = split(readline(file))
        popfirst!(col)
        a[i] = parse.(Float64, col)
    end

    readline(file)
    readline(file)
    readline(file)
    readline(file)
    ## reciprocal lattice vectors
    b = Vector{SVector{3,Float64}}(undef, 3)
    for i in 1:3
        col = split(readline(file))
        popfirst!(col)
        b[i] = parse.(Float64, col)
    end

    # projections
    # k-point grid
    # main
    # wannierise
    # disentangle
    # plotting
    # k-mesh
    # etc...

    return a, b
end

parse_sym(filename) = open(filename) do file
    nsymmetry = parse(Int, readline(file))
    readline(file)
    syms = Vector{AffineTransform{3,Float64}}(undef, nsymmetry)
    S = Matrix{Float64}(undef, (3,4))
    for i in 1:nsymmetry
        for j in 1:4
            col = split(readline(file))
            S[:,j] .= parse.(Float64, col)
        end
        syms[i] = AffineTransform(S[:,1:3], S[:,4])
        readline(file)
    end
    return nsymmetry, syms
end

"""
    load_wannier90_data(seedname; velocity_kind=:none, read_pos_op=true, compact=:N)

Reads Wannier90 output files with the given `seedname` to return the Hamiltonian
(optionally with band velocities if `velocity_kind` is specified, and orbital
velocities if `read_pos_op` is true) and the full Brillouin zone limits. The
keyword `compact` is available if to compress the Fourier series if its Fourier
coefficients are known to be Hermitian.
"""
function load_wannier90_data(seedname::String; read_sym=false, read_pos_op=true, velocity_kind=:none, compact=:N)
    a, b = try
        parse_wout(seedname * ".wout")
    catch
        @info "Couldn't find $(seedname).wout so using Cartesian unit lattice instead"
        (SVector{3,Float64}[[1,0,0], [0,1,0], [0,0,1]], SVector{3,Float64}[[2π,0,0], [0,2π,0], [0,0,2π]])
    end

    periods = ntuple(i -> norm(b[i]), Val{3}())
    HV = if velocity_kind == :none
        load_hamiltonian(seedname * "_hr.dat"; period=periods, compact=compact)
    else
        try
            read_pos_op || error()
            @info "Using orbital velocities"
            load_hamiltonian_velocities(seedname * "_hr.dat", seedname * "_r.dat"; period=periods, compact=compact, kind=velocity_kind)
        catch
            @info "Using band velocities"
            load_hamiltonian_velocities(seedname * "_hr.dat"; period=periods, compact=compact, kind=velocity_kind)
        end
    end

    BZ = try
        read_sym || error()
        @info "IBZ not implemented automatically -- construct one manually from FBZ"
        error()
        #=
        nsymmetry, syms = parse_sym(seedname * ".sym")
        @info "Using irreducible Brillouin zone"
        IrreducibleBZ(a, b, syms)
        =#
    catch
        @info "Using full Brillouin zone"
        FullBZ(a, b)
    end

    return HV, BZ
end