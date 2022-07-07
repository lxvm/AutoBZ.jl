# this file supplies custom methods for Hermitian representations of matrices

"""
Calculate the trace of a Hermitian matrix (possibly in compressed format)
"""
htr(A::AbstractMatrix) = tr(A)
htr(A::SVector{6}) = @inbounds +(A[1], A[4], A[6])

"""
Calculate the inverse of a Hermitian matrix using its lower triangle
"""
function hinv(A::AbstractMatrix{T})::SMatrix{3,3,promote_type(Float64, T)} where {T<:Number}
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
    SMatrix{3,3}(b1, b2, b3, conj(b2), b5, b6, conj(b3), conj(b6), b9)
end

"""
Calculate the inverse of a Hermitian matrix in compressed format
"""
function hinv(A::SVector{6,T})::SVector{6,promote_type(Float64, T)} where {T<:Number}
    @inbounds x0 = A[4]*A[6] - abs2(A[5])
    @inbounds x1 = A[5]*conj(A[3]) - conj(A[2])*A[6]
    @inbounds x2 = conj(A[2]*A[5]) - A[4]*conj(A[3])
    @inbounds idet = inv(A[1]*x0 + A[2]*x1 + A[3]*x2)
    @inbounds b1 = x0*idet
    @inbounds b2 = (conj(A[5])*A[3] - A[6]*A[2])*idet
    @inbounds b3 = (A[2]*A[5]-A[3]*A[4])*idet
    @inbounds b4 = (A[6]*A[1] - abs2(A[3]))*idet
    @inbounds b5 = (conj(A[2])*A[3] - A[1]*A[5])*idet
    @inbounds b6 = (A[1]*A[4] - abs2(A[2]))*idet
    SVector{6,promote_type(Float64, T)}(b1, b2, b3, b4, b5, b6)
end