function pythtb2fourier(m)
    @assert m._dim_k == m._dim_r
    @assert m._dim_k >= 1
    @assert m._nspin == 1 "Spin case not implemented"

    dim_r = m._dim_r
    Rmax = zeros(Int, dim_r)
    for h in eachrow(m._hoppings)
        r = h[4]
        for i in 1:dim_r
            if Rmax[i] < abs(r[i])
                Rmax[i] = abs(r[i])
            end
        end
    end
    offset = CartesianIndex(Rmax...)
    offset = -offset -one(offset)

    norb = m._norb
    C = zeros(ComplexF64, norb, norb, (2Rmax .+ 1)...)
    for h in eachrow(m._hoppings)
        amp, _i, _j, _r = h
        r = CartesianIndex(_r...)
        C[_i+1,_j+1, r-offset] += amp
        C[_j+1,_i+1,-r-offset] += conj(amp)
    end
    _C = Array{SMatrix{norb,norb,ComplexF64,norb^2},dim_r}(undef, size(C)[3:end]...)
    for i in CartesianIndices(_C)
        _C[i] = view(C, :, :, i)
    end
    _C[-offset] += Diagonal(m._site_energies)
    return FourierSeries(_C; period=1.0, offset=Tuple(offset))
end

function pythtb2hamiltonian(m; kws...)
    f = pythtb2fourier(m)
    return HamiltonianInterp(AutoBZ.Freq2RadSeries(HermitianFourierSeries(f)); kws...)
end

pythtb2interp(::Type{<:HamiltonianInterp}, m; kws...) = pythtb2hamiltonian(m; kws...)
function pythtb2interp(::Type{<:GradientVelocityInterp}, m; kws...)
    GradientVelocityInterp(pythtb2hamiltonian(m), m._lat'; kws...)
end
function pythtb2interp(::Type{<:CovariantVelocityInterp}, m; kws...)
    throw(ArgumentError("CovariantVelocityInterp not supported for pythtb since position operator can't be Wannier interpolated"))
end

function pythtb2bz(bz::AbstractBZ, m, species)
    A = m._lat'
    return load_bz(bz, A)
end

function pythtb2bz(bz::IBZ, m, species)
    A = m._lat'
    B = AutoBZCore.canonical_reciprocal_basis(A)
    positions = m._orb'
    return load_bz(bz, A, B, species, positions)
end

"""
    load_pythtb_data(m, [species]; bz=FBZ(), interp=HamiltonianInterp, kws...)

Returns `(interp, bz)` from a PythTB model `m` for use with AutoBZ.jl solvers.
Only supports spinless, fully-periodic systems.
Supplying a list of atomic `species` is helpful when chosing a `bz` of [`AutoBZCore.IBZ`](@extref),
since PythTB loses some information by combining the atomic and orbital indices into a
multi-index. By default, `species` assigns a different element to orbitals at different positions.
To call PythTB from Julia see e.g. [PyCall.jl](https://github.com/JuliaPy/PyCall.jl).
"""
function load_pythtb_data(m, species=_default_species(m); bz=FBZ(), interp=HamiltonianInterp, kws...)
    return pythtb2interp(interp, m; kws...), pythtb2bz(bz, m, species)
end

function _default_species(m)
    positions = unique(eachrow(m._orb))
    return [findfirst(isequal(x), positions) for x in eachrow(m._orb)]
end
