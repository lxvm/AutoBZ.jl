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

parse_self_energy_scalar(filename, ::Type{F}) where {F<:AbstractFloat} = open(filename) do file
    nfpts = parse(Int, readline(file))
    omegas = Vector{F}(undef, nfpts)
    values = Vector{Complex{F}}(undef, nfpts)
    for i in 1:nfpts
        col = split(readline(file))
        omegas[i] = parse(F, col[1])
        re = parse(F, col[2])
        im = parse(F, col[3])
        values[i] = complex(re, im)
    end
    return nfpts, omegas, values
end

parse_self_energy_diagonal(filename, ::Type{F}) where {F<:AbstractFloat} = open(filename) do file
    nfpts = parse(Int, readline(file))
    num_wann = parse(Int, readline(file))
    omegas = Vector{F}(undef, nfpts)
    values = Vector{SVector{num_wann,Complex{F}}}(undef, nfpts)
    omega = Vector{F}(undef, num_wann)
    value = Vector{Complex{F}}(undef, num_wann)
    for i in 1:nfpts
        for j in 1:num_wann
            col = split(readline(file))
            omega[j] = parse(F, col[1])
            n = parse(Int, col[2])
            re = parse(F, col[3])
            im = parse(F, col[4])
            value[n] = complex(re, im)
        end
        omegas[i] = only(unique(omega))
        values[i] = SVector{num_wann,Complex{F}}(value)
    end
    return nfpts, num_wann, omegas, values
end

parse_self_energy_matrix(filename, ::Type{F}) where {F<:AbstractFloat} = open(filename) do file
    nfpts = parse(Int, readline(file))
    num_wann = parse(Int, readline(file))
    omegas = Vector{F}(undef, nfpts)
    values = Vector{SMatrix{num_wann,num_wann,Complex{F},num_wann^2}}(undef, nfpts)
    omega = Vector{F}(undef, num_wann^2)
    value = Matrix{Complex{F}}(undef, num_wann, num_wann)
    for i in 1:nfpts
        for j in 1:num_wann^2
            col = split(readline(file))
            omega[j] = parse(F, col[1])
            m = parse(Int, col[2])
            n = parse(Int, col[3])
            re = parse(F, col[4])
            im = parse(F, col[5])
            value[m,n] = complex(re, im)
        end
        omegas[i] = only(unique(omega))
        values[i] = SMatrix{num_wann,num_wann,Complex{F},num_wann^2}(value)
    end
    return nfpts, num_wann, omegas, values
end

"""
    load_self_energy(filename; [sigdigits=8, output=:interp, degree=:default])

Read the self energy data in `filename`, which should be in either `:scalar`,
`:diagonal`, or `:matrix` format, and return a self-energy evaluator. Note that
the frequency data is assumed to be an equispace grid. The optional argument
`degree` indicates the degree of barycentric Lagrange interpolation, and that
`sigdigits` indicates the number of significant digits used to round the
frequency data so as to avoid rounding errors.

The keyword `output` may take values of `:interp` (default) or `:raw` which will
either return the self energy data wrapped with a high-order interpolating
function (details below) or in the latter option, just the raw data.

The interpolating function will always be a piecewise high-order polynomial
whose details depend on the distribution of frequency points:
* For equispaced frequency points, a local barycentric Lagrange interpolant is
  used with a default polynomial degree of 8
* For other frequency point distributions, first a global rational approximant
  is formed using the [AAA algorithm](https://arxiv.org/abs/1612.00337), and
  then h-adaptive Chebyshev interpolation is performed on the global interpolant
  in order to obtain a fast-to-evaluate representation of default polynomial
  degree 16.
"""
function load_self_energy(filename; precision=Float64, sigdigits=8, output=:interp, degree=:default, tol=1e-13, mmax=100)
    fmt = get_self_energy_format(filename)
    if fmt == :scalar
        nfpts, omegas, values = parse_self_energy_scalar(filename, precision)
    elseif fmt == :diagonal
        nfpts, num_wann, omegas, values = parse_self_energy_diagonal(filename, precision)
    elseif fmt == :matrix
        nfpts, num_wann, omegas, values = parse_self_energy_matrix(filename, precision)
    else
        throw(ErrorException("self energy format not implemented"))
    end
    if output == :raw
        return omegas, values
    else
        interpolant = try
            deg = degree == :default ? 8 : degree
            construct_lagrange(omegas, values, sigdigits, deg)
        catch
            order = degree == :default ? 16 : degree
            construct_chebyshev(omegas, values, order; tol=tol, mmax=mmax)
        end
        a, b = extrema(omegas)
        return if fmt == :scalar
            ScalarSelfEnergy(interpolant, a,b)
        elseif fmt == :diagonal
            DiagonalSelfEnergy(interpolant, a,b)
        elseif fmt == :matrix
            MatrixSelfEnergy(interpolant, a,b)
        end
    end
end

function construct_lagrange(omegas, values, sigdigits, degree)
    LocalEquiBaryInterp(round.(omegas; sigdigits=sigdigits), values, degree=degree)
end

function construct_chebyshev(omegas, values::Vector{<:Number}, order; atol=1e-6, tol=1e-13, mmax=100)
    interp = aaa(omegas, values; tol=tol, mmax=mmax)
    hchebinterp(interp, extrema(omegas)...; order=order, atol=atol)
end
function construct_chebyshev(omegas, values::Vector{T}, order; atol=1e-6, tol=1e-13, mmax=100) where {T<:SArray}
    interp = ntuple(n -> aaa(omegas, getindex.(values, n); tol=tol, mmax=mmax), length(T))
    hchebinterp(x -> T(map(f -> f(x), interp)), extrema(omegas)...; order=order, atol=atol)
end
