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
    load_hamiltonian(filename; period=1.0)

Load an ab-initio Hamiltonian output from Wannier90 into `filename` as an
evaluatable `FourierSeries` whose periodicity can be set by the keyword argument
`period` which defaults to setting the period along each dimension to `1.0`. To
define different periods for different dimensions, pass an `SVector` as the
`period`.
"""
function load_hamiltonian(filename; period=1.0)
    date_time, num_wann, nrpts, degen, irvec, C_ = parse_hamiltonian(filename)
    s = Int(cbrt(nrpts))
    r = Int((s-1)/2)
    C = reshape(similar(C_), s, s, s)
    for (i, idx) in enumerate(irvec)
        C[CartesianIndex((idx .+ (r+1))...)] = C_[i]
    end
    FourierSeries(OffsetArray(C, -r:r, -r:r, -r:r), to_3period(period))
end
to_3period(x::T) where {T} = fill(x, SVector{3,T})
to_3period(x::SVector{3}) = identity(x)

"""
    parse_hamiltonian(filename)

Parse an ab-initio Hamiltonian output from Wannier90 into `filename`, extracting
the fields `(date_time, num_wann, nrpts, degen, irvec, C)`
"""
parse_gauge_transform(filename) = open(filename) do file
    date_time = readline(file)

    num_wann = parse(Int, readline(file))
    nrpts = parse(Int, readline(file))

    X = Vector{SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}}(undef, nrpts)
    Y = Vector{SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}}(undef, nrpts)
    Z = Vector{SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}}(undef, nrpts)
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
            re = parse(Float64, col[6])
            im = parse(Float64, col[7])
            x[m,n] = complex(re, im)
            re = parse(Float64, col[8])
            im = parse(Float64, col[9])
            y[m,n] = complex(re, im)
            re = parse(Float64, col[10])
            im = parse(Float64, col[11])
            z[m,n] = complex(re, im)
        end
        X[k] = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}(x)
        Y[k] = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}(y)
        Z[k] = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}(z)
    end
    date_time, num_wann, nrpts, irvec, X, Y, Z
end


"""
    load_gauge_transform(filename; period=1.0)

Load an ab-initio Hamiltonian output from Wannier90 into `filename` as an
evaluatable `FourierSeries` whose periodicity can be set by the keyword argument
`period` which defaults to setting the period along each dimension to `1.0`. To
define different periods for different dimensions, pass an `SVector` as the
`period`.
"""
function load_gauge_transform(filename; period=1.0)
    date_time, num_wann, nrpts, irvec, X_, Y_, Z_ = parse_gauge_transform(filename)
    s = Int(cbrt(nrpts))
    r = Int((s-1)/2)
    X = reshape(similar(X_), s, s, s)
    Y = reshape(similar(Y_), s, s, s)
    Z = reshape(similar(Z_), s, s, s)
    for (i, idx) in enumerate(irvec)
        X[CartesianIndex((idx .+ (r+1))...)] = X_[i]
        Y[CartesianIndex((idx .+ (r+1))...)] = Y_[i]
        Z[CartesianIndex((idx .+ (r+1))...)] = Z_[i]
    end
    periods = to_3period(period)
    FX = FourierSeries(OffsetArray(X, -r:r, -r:r, -r:r), periods)
    FY = FourierSeries(OffsetArray(Y, -r:r, -r:r, -r:r), periods)
    FZ = FourierSeries(OffsetArray(Z, -r:r, -r:r, -r:r), periods)
    ManyFourierSeries(FX, FY, FZ)
end