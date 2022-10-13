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