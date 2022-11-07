function LinearAlgebra.inv(A::SHermitianCompact{3,T,6}) where {T<:Real}
    F = Base.promote_op(inv,T)
    SHermitianCompact{3,F,6}(SVector{6,F}(hinv(A)))
end

function LinearAlgebra.inv(A::SHermitianCompact{3,T,6}) where T
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
