get_self_energy_format(filename) = open(filename) do file
    col = split(readline(file))
    while length(col) == 1
        col = split(readline(file))
    end
    if (len = length(col)) == 3
        return :scalar
    elseif len == 4
        return :diagonal
    elseif len == 5
        return :matrix
    else
        throw(ErrorException("format unrecognized"))
    end
end

parse_self_energy_scalar(filename) = open(filename) do file
    nfpts = parse(Int, readline(file))
    omegas = Vector{Float64}(undef, nfpts)
    values = Vector{ComplexF64}(undef, nfpts)
    for i in 1:nfpts
        col = split(readline(file))
        omegas[i] = parse(Float64, col[1])
        re = parse(Float64, col[2])
        im = parse(Float64, col[3])
        values[i] = complex(re, im)
    end
    return nfpts, omegas, values
end

parse_self_energy_diagonal(filename) = open(filename) do file
    nfpts = parse(Int, readline(file))
    num_wann = parse(Int, readline(file))
    omegas = Vector{Float64}(undef, nfpts)
    values = Vector{Diagonal{ComplexF64,SVector{num_wann,ComplexF64}}}(undef, nfpts)
    omega = Vector{Float64}(undef, num_wann)
    value = Vector{ComplexF64}(undef, num_wann)
    for i in 1:nfpts
        for j in 1:num_wann
            col = split(readline(file))
            omega[j] = parse(Float64, col[1])
            n = parse(Int, col[2])
            re = parse(Float64, col[3])
            im = parse(Float64, col[4])
            value[n] = complex(re, im)
        end
        omegas[i] = only(unique(omega))
        values[i] = Diagonal(SVector{num_wann,ComplexF64}(value))
    end
    return nfpts, num_wann, omegas, values
end

parse_self_energy_matrix(filename) = open(filename) do file
    nfpts = parse(Int, readline(file))
    num_wann = parse(Int, readline(file))
    omegas = Vector{Float64}(undef, nfpts)
    values = Vector{SMatrix{num_wann,numwann,ComplexF64,num_wann^2}}(undef, nfpts)
    omega = Vector{Float64}(undef, num_wann^2)
    value = Matrix{ComplexF64}(undef, num_wann, num_wann)
    for i in 1:nfpts
        for j in 1:num_wann^2
            col = split(readline(file))
            omega[j] = parse(Float64, col[1])
            m = parse(Int, col[2])
            n = parse(Int, col[3])
            re = parse(Float64, col[4])
            im = parse(Float64, col[5])
            value[m,n] = complex(re, im)
        end
        omegas[i] = only(unique(omega))
        values[i] = SMatrix{num_wann,num_wann,ComplexF64,num_wann^2}(value)
    end
    return nfpts, num_wann, omegas, values
end

"""
    load_self_energy(filename, [degree=8, sigdigits=8])

Read the self energy data in `filename`, which should be in either `:scalar`,
`:diagonal`, or `:matrix` format, and return a self-energy evaluator. Note that
the frequency data is assumed to be an equispace grid. The optional argument
`degree` indicates the degree of barycentric Lagrange interpolation, and that
`sigdigits` indicates the number of significant digits used to round the
frequency data so as to avoid rounding errors.
"""
function load_self_energy(filename; sigdigits=8)
    fmt = get_self_energy_format(filename)
    if fmt == :scalar
        nfpts, omegas, values = parse_self_energy_scalar(filename)
    elseif fmt == :diagonal
        nfpts, num_wann, omegas, values = parse_self_energy_diagonal(filename)
    elseif fmt == :matrix
        nfpts, num_wann, omegas, values = parse_self_energy_matrix(filename)
    else
        throw(ErrorException("self energy format not implemented"))
    end
    return round.(omegas; sigdigits=sigdigits), values
end