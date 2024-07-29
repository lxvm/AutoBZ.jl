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

"""
    pythtb2hamiltonian(m)

Converts a PythTB model `m` to a [`HamiltonianInterp`](@ref).
Only supports spinless, fully-periodic systems.
To call PythTB from Julia see e.g. [PyCall.jl](https://github.com/JuliaPy/PyCall.jl).
"""
function pythtb2hamiltonian(m; kws...)
    f = pythtb2fourier(m)
    return HamiltonianInterp(AutoBZ.Freq2RadSeries(HermitianFourierSeries(f)); kws...)
end
