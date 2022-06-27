"""
    Integrate a function on an equispace grid with the same number of grid
    points along each dimension
"""
function equispace_integration(f, p::Int; rtol=sqrt(eps()))
    r = zero(SMatrix{3,3,ComplexF64})
    x = range(0.0, step=inv(p), length=p)
    for k in 1:p
        for j in 1:p
            for i in 1:p
                @inbounds r += f(SVector(x[i], x[j], x[k]))
            end
        end
    end
    r*inv(p)^3
end

function equispace_integration(f::FourierSeries{3}, p::Int)
    r = Array{SMatrix{3,3,ComplexF64}}(undef, p^3)
    x = range(0.0, step=inv(p), length=p)
    for k in 1:p
        for j in 1:p
            for i in 1:p
                @inbounds r[i,j,k] = f(SVector(x[i], x[j], x[k]))
            end
        end
    end
    r
end