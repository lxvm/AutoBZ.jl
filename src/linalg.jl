# method to intercept for units
_inv(args...; kws...) = inv(args...; kws...)
_eigen(args...; kws...) = eigen(args...; kws...)

function _inv(A::SHermitianCompact{3,T,6}) where {T<:Real}
    F = Base.promote_op(inv,T)
    SHermitianCompact{3,F,6}(SVector{6,F}(hinv(A)))
end

function _inv(A::SHermitianCompact{3,T,6}) where T
    SMatrix{3,3,Base.promote_op(inv,T),9}(hinv(A))
end

"""
    hinv(A::SHermitianCompact{3})

Calculate the inverse of a `SHermitianCompact` matrix using its lower triangle.
Note that if the elements on the diagonal are complex, the inverse is not
Hermitian.
"""
hinv(A::SHermitianCompact{3,T,6}) where T = hinv_(A.lowertriangle...)
function hinv_(A1::T, A2::T, A3::T, A5::T, A6::T, A9::T) where {T<:Real}
    x0 = A5*A9 - A6*A6
    x1 = A6*A3 - A2*A9
    x2 = A2*A6 - A5*A3
    idet = inv(A1*x0 + A2*x1 + A3*x2)
    b1 = x0*idet
    b2 = (A6*A3 - A9*A2)*idet
    b3 = (A2*A6 - A3*A5)*idet
    b5 = (A9*A1 - A3*A3)*idet
    b6 = (A2*A3 - A1*A6)*idet
    b9 = (A1*A5 - A2*A2)*idet
    b1, b2, b3, b5, b6, b9
end
function hinv_(A1::T, A2::T, A3::T, A5::T, A6::T, A9::T) where T
    x0 = A5*A9 - abs2(A6)
    x1 = A6*conj(A3) - conj(A2)*A9
    x2 = conj(A2*A6) - A5*conj(A3)
    idet = inv(A1*x0 + A2*x1 + A3*x2)
    b1 = x0*idet
    b2 = (conj(A6)*A3 - A9*A2)*idet
    b3 = (A2*A6-A3*A5)*idet
    b4 = x1*idet
    b5 = (A9*A1 - abs2(A3))*idet
    b6 = (conj(A2)*A3 - A1*A6)*idet
    b7 = x2*idet
    b8 = (A2*conj(A3) - A1*conj(A6))*idet
    b9 = (A1*A5 - abs2(A2))*idet
    b1, b2, b3, b4, b5, b6, b7, b8, b9
end

"""
    diag_inv(A)

Calculate the diagonal entries of the inverse of `A`.
"""
diag_inv(A) = diag(_inv(A)) # fallback method
diag_inv(A::SMatrix{3,3,T}) where T = SVector{3,T}(diag_inv_(A.data...))
function diag_inv_(A1::T, A2::T, A3::T, A4::T, A5::T, A6::T, A7::T, A8::T, A9::T) where T
    x0 = A5*A9 - A6*A8
    x1 = A6*A7 - A4*A9
    x2 = A4*A8 - A5*A7
    idet = inv(A1*x0 + A2*x1 + A3*x2)
    (x0, A9*A1 - A7*A3, A1*A5 - A2*A4).*idet
end
diag_inv(A::SHermitianCompact{3,T}) where T = SVector{3,T}(diag_inv_(A.lowertriangle...))
function diag_inv_(A1::T, A2::T, A3::T, A5::T, A6::T, A9::T) where T
    x0 = A5*A9 - abs2(A6)
    x1 = A6*conj(A3) - conj(A2)*A9
    x2 = conj(A2*A6) - A5*conj(A3)
    idet = inv(A1*x0 + A2*x1 + A3*x2)
    (x0, A9*A1 - abs2(A3), A1*A5 - abs2(A2)).*idet
end


"""
    tr_inv(A)

Calculate the trace of the inverse of `A`.
"""
tr_inv(A) = tr(_inv(A))

tr_inv(A::SMatrix{3,3}) = tr_inv_(A.data...)
function tr_inv_(A1::T, A2::T, A3::T, A4::T, A5::T, A6::T, A7::T, A8::T, A9::T) where T
    x0 = A5*A9 - A6*A8
    x1 = A6*A7 - A4*A9
    x2 = A4*A8 - A5*A7
    idet = inv(A1*x0 + A2*x1 + A3*x2)
    +(x0, A9*A1 - A7*A3, A1*A5 - A2*A4)*idet
end


tr_inv(A::SHermitianCompact{3}) = tr_inv_(A.lowertriangle...)
function tr_inv_(A1::T, A2::T, A3::T, A5::T, A6::T, A9::T) where T
    x0 = A5*A9 - abs2(A6)
    x1 = A6*conj(A3) - conj(A2)*A9
    x2 = conj(A2*A6) - A5*conj(A3)
    idet = inv(A1*x0 + A2*x1 + A3*x2)
    +(x0, A9*A1 - abs2(A3), A1*A5 - abs2(A2))*idet
end

"""
    tr_mul(A, B)

Calculate `tr(A*B)` without storing the intermediate result.
"""
@inline tr_mul(A, B) = tr(A*B) # this gets optimized by the compiler

"""
    tr_kron(A::T, B::T) where {T<:SVector{AbstractMatrix}}

Returns a matrix whose `[i,j]`th entry is `tr(A[i]*B[j])`.
"""
@inline function tr_kron(A::SVector{N,TA}, B::SVector{N,TB}) where {N,TA,TB}
    T = Base.promote_op(tr_mul, TA, TB)
    data = ntuple(Val(N^2)) do n
        d, r = divrem(n-1, N)
        tr_mul(A[r+1], B[d+1])
    end
    SMatrix{N,N,T,N^2}(data)
end

"""
    herm(A::AbstractMatrix)

Return the Hermitian part of the matrix `A`, i.e. `(A+A')/2`.
"""
@inline herm(A) = (convert(eltype(A), 1//2)*I) * (A + A')

"""
    commutator(A, B)

Return the commutator `[A, B] = A*B - B*A`.
"""
@inline commutator(A, B) = A*B - B*A

function isapproxhermitian(A::AbstractMatrix; kwargs...)
    indsm, indsn = axes(A)
    if indsm != indsn
        return false
    end
    for i = indsn, j = i:last(indsn)
        if !isapprox(A[i,j], adjoint(A[j,i]); kwargs...)
            return false
        end
    end
    return true
end
