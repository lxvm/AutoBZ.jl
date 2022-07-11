function LinearAlgebra.inv(A::SHermitianCompact{3,T,6}) where {T}
    b1, b2, b3, b5, b6, b9 = _hinv(A)
    SHermitianCompact{3,Base.promote_op(inv,T),6}(b1, b2, b3, nothing, b5, b6, nothing, nothing, b9)
end

"""
Calculate the inverse of a Hermitian matrix using its lower triangle
"""
function _hinv(A::StaticMatrix{3,3})
    @inbounds x0 = A[5]*A[9] - abs2(A[6])
    @inbounds x1 = A[6]*conj(A[3]) - conj(A[2])*A[9]
    @inbounds x2 = conj(A[2]*A[6]) - A[5]*conj(A[3])
    @inbounds idet = inv(A[1]*x0 + A[2]*x1 + A[3]*x2)
    @inbounds b1 = x0*idet
    @inbounds b2 = (conj(A[6])*A[3] - A[9]*A[2])*idet
    @inbounds b3 = (A[2]*A[6]-A[3]*A[5])*idet
    @inbounds b5 = (A[9]*A[1] - abs2(A[3]))*idet
    @inbounds b6 = (conj(A[2])*A[3] - A[1]*A[6])*idet
    @inbounds b9 = (A[1]*A[5] - abs2(A[2]))*idet
    b1, b2, b3, b5, b6, b9
end