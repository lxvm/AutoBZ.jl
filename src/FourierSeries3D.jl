"""
    FourierSeries3D(f::Array{T,3}; period=2pi, deriv=0, offset=0, shift=0)

This type is an `AbstractInplaceFourierSeries{3}` and unlike `FourierSeries` is
specialized for 3D Fourier series and does not allocate a new array every time
`contract` is called on it. This type stores the intermediate arrays used in a
calculation and assumes that the size of `f` on each axis is odd because it
treats the zeroth harmonic as the center of the array (i.e. `(size(f) .÷ 2) .+
1`).
"""
struct FourierSeries3D{T,K,A,O,Q} <: AbstractInplaceFourierSeries{3,T}
    # fi where i represents array dims
    f3::Array{T,3}
    f2::Array{T,2}
    f1::Array{T,1}
    f0::Array{T,0}
    k::NTuple{3,K}
    a::NTuple{3,A}
    o::NTuple{3,O}
    q::NTuple{3,Q}
end

function FourierSeries3D(f3_::Array{T_,3}; period=2pi, deriv=Val(0), offset=0, shift=0.0) where T_
    T = fourier_type(T_, eltype(period))
    f3 = convert(Array{T,3}, f3_)
    f2 = Array{T,2}(undef, size(f3,1), size(f3, 2))
    f1 = Array{T,1}(undef, size(f3,1))
    f0 = Array{T,0}(undef)
    period = fill_ntuple(period, 3)
    deriv  = fill_ntuple(deriv,  3)
    offset = fill_ntuple(offset, 3)
    shift  = fill_ntuple(shift,  3)
    FourierSeries3D(f3, f2, f1, f0, 2pi ./ period, deriv, offset, shift)
end

period(f::FourierSeries3D) = 2pi ./ f.k
@generated function contract!(f::FourierSeries3D, x::Number, ::Val{dim}) where dim
    quote
        fourier_contract!(f.$(Symbol(:f, dim-1)), f.$(Symbol(:f, dim)), x-f.q[$dim], f.k[$dim], f.a[$dim], f.o[$dim], Val($dim))
        return f
    end
end

evaluate(f::FourierSeries3D, x::NTuple{1}) =
    fourier_evaluate(f.f1, x[1]-f.q[1], f.k[1], f.a[1], f.o[1])
evaluate(f::FourierSeries3D, x::NTuple{N}) where N =
    evaluate(contract(f, x[N], Val(N)), x[1:N-1])
evaluate(f::FourierSeries3D, x) =
    evaluate(f, promote(x...))


@testset "FourierSeries3D" begin
    d=3; nxtest=5
    n = 11; m = div(n,2)
    for T in (ComplexF64, SMatrix{5,5,ComplexF64,25})
        C = rand(T, ntuple(_->n, d)...)
        OC = OffsetArray(C, ntuple(_->-m:m, d)...)
        for _ in 1:nxtest
            x = rand(d)
            # test period
            periods = rand(d)
            f = FourierSeries3D(C, period=periods)
            @test all(period(f) .≈ periods)
            @test f(x) ≈ ref_evaluate(C, x, 2pi ./ periods)
            # test derivative
            for (deriv, a) in ((Val(0), 0), (Val(1), 1), fill(rand(1:4, d), 2))
                f = FourierSeries3D(C, period=1, deriv=a)
                @test f(x) ≈ ref_evaluate(C, x, 2pi, a)
            end
            # test offset
            f = FourierSeries3D(C, period=1, offset=-m-1)
            @test f(x) ≈ ref_evaluate(OC, x, 2pi)
            # test shift
            q = rand(d)
            f = FourierSeries3D(C, period=1, shift=q)
            @test f(x) ≈ ref_evaluate(C, x-q, 2pi)
        end
    end
end
