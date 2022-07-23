function equispace_integration(f, p::Int, ::TetrahedralLimits)
    equispace_integration(f, p, cubic_ibz(p)...)
    # for (x, w) in get_x_w(p)
    #     r += w*f(x)
    # end
end
function equispace_integration(f, p::Int, flag::Array{Bool,3}, wsym::Vector{Int})
    r = zero(eltype(f))
    x = range(0, step=inv(p), length=p)
    cnt = 0
    for k in 1:p
        any(flag[:, :, k]) || continue
        @inbounds g = contract(f, x[k])
        for j in 1:p
            any(flag[:, j, k]) || continue
            @inbounds h = contract(g, x[j])
            for i in 1:p
                if flag[i,j,k]
                    cnt += 1
                    @inbounds r += wsym[cnt]*h(SVector(x[i]))
                end
            end
        end
    end
    r*inv(p)^3
end

function evaluate_series_ibz!(r, f::FourierSeries{3}, p::Int, flag::Array{Bool,3})
    x = range(0, step=inv(p), length=p)
    cnt = 0
    for k in 1:p
        any(flag[:, :, k]) || continue
        @inbounds g = contract(f, x[k])
        for j in 1:p
            any(flag[:, j, k]) || continue
            @inbounds h = contract(g, x[j])
            for i in 1:p
                if flag[i,j,k]
                    cnt += 1
                    @inbounds r[cnt] = h(SVector(x[i]))
                end
            end
        end
    end
    r
end

get_x_w(p) = ijk_to_x(p, cubic_ibz(p)...)

function ijk_to_x(p::Int, flag, wsym)
    x = range(0, step=inv(p), length=p)
    ((SVector(x[i[1]], x[i[2]], x[i[3]]), wsym[j]) for (j, i) in enumerate(CartesianIndices(flag)[flag]))
end

"""
Generate the equispace grid indices with associated weights for the tetrahedral
irreducible BZ of a cubic full BZ
"""
function cubic_ibz(p::Int)
    xsym = MMatrix{3,48}(zeros(3, 48))
    x = range(-1.0, step=2inv(p), length=p)
    flag = ones(Bool, (p,p,p))
    nsym = 0
    wsym = zeros(Int, p^3)
    for k in 1:p
        for j in 1:p
            for i in 1:p
                @inbounds flag[i,j,k] || continue
                @inbounds symmetrize!(SVector(x[i], x[j], x[k]), xsym)
                nsym += 1
                wsym[nsym] = 1
                for l in 2:48
                    @inbounds ii, jj, kk = 0.5p .* (xsym[:,l] .+ 1) .+ 1
                    (round(Int, ii) - ii) > 1e-12 && throw("Inexact index")
                    (round(Int, jj) - jj) > 1e-12 && throw("Inexact index")
                    (round(Int, kk) - kk) > 1e-12 && throw("Inexact index")
                    ii = round(Int, ii)
                    jj = round(Int, jj)
                    kk = round(Int, kk)
                    any((ii>p, jj>p, kk>p)) && continue
                    all((ii==i,jj==j,kk==k)) && continue
                    @inbounds if flag[ii,jj,kk]
                        @inbounds flag[ii,jj,kk] = false
                        @inbounds wsym[nsym] += 1
                    end
                end
            end
        end
    end
    flag, wsym
end

"""
Generate the images of a point in the cube [-1,1]^3 under the automorphism group
of the cube
"""
function symmetrize!(x::SVector{3}, xsym::MMatrix{3,48})
    @inbounds xsym[:, 1] = [ x[1], x[2], x[3]]
    @inbounds xsym[:, 2] = [-x[1], x[2], x[3]]
    @inbounds xsym[:, 3] = [ x[1],-x[2], x[3]]
    @inbounds xsym[:, 4] = [ x[1], x[2],-x[3]]
    @inbounds xsym[:, 5] = [-x[1],-x[2], x[3]]
    @inbounds xsym[:, 6] = [-x[1], x[2],-x[3]]
    @inbounds xsym[:, 7] = [ x[1],-x[2],-x[3]]
    @inbounds xsym[:, 8] = [-x[1],-x[2],-x[3]]
   
    @inbounds xsym[:, 9] = [ x[2], x[1], x[3]]
    @inbounds xsym[:,10] = [-x[2], x[1], x[3]]
    @inbounds xsym[:,11] = [ x[2],-x[1], x[3]]
    @inbounds xsym[:,12] = [ x[2], x[1],-x[3]]
    @inbounds xsym[:,13] = [-x[2],-x[1], x[3]]
    @inbounds xsym[:,14] = [-x[2], x[1],-x[3]]
    @inbounds xsym[:,15] = [ x[2],-x[1],-x[3]]
    @inbounds xsym[:,16] = [-x[2],-x[1],-x[3]]
   
    @inbounds xsym[:,17] = [ x[3], x[2], x[1]]
    @inbounds xsym[:,18] = [-x[3], x[2], x[1]]
    @inbounds xsym[:,19] = [ x[3],-x[2], x[1]]
    @inbounds xsym[:,20] = [ x[3], x[2],-x[1]]
    @inbounds xsym[:,21] = [-x[3],-x[2], x[1]]
    @inbounds xsym[:,22] = [-x[3], x[2],-x[1]]
    @inbounds xsym[:,23] = [ x[3],-x[2],-x[1]]
    @inbounds xsym[:,24] = [-x[3],-x[2],-x[1]]
   
    @inbounds xsym[:,25] = [ x[1], x[3], x[2]]
    @inbounds xsym[:,26] = [-x[1], x[3], x[2]]
    @inbounds xsym[:,27] = [ x[1],-x[3], x[2]]
    @inbounds xsym[:,28] = [ x[1], x[3],-x[2]]
    @inbounds xsym[:,29] = [-x[1],-x[3], x[2]]
    @inbounds xsym[:,30] = [-x[1], x[3],-x[2]]
    @inbounds xsym[:,31] = [ x[1],-x[3],-x[2]]
    @inbounds xsym[:,32] = [-x[1],-x[3],-x[2]]
   
    @inbounds xsym[:,33] = [ x[2], x[3], x[1]]
    @inbounds xsym[:,34] = [-x[2], x[3], x[1]]
    @inbounds xsym[:,35] = [ x[2],-x[3], x[1]]
    @inbounds xsym[:,36] = [ x[2], x[3],-x[1]]
    @inbounds xsym[:,37] = [-x[2],-x[3], x[1]]
    @inbounds xsym[:,38] = [-x[2], x[3],-x[1]]
    @inbounds xsym[:,39] = [ x[2],-x[3],-x[1]]
    @inbounds xsym[:,40] = [-x[2],-x[3],-x[1]]
   
    @inbounds xsym[:,41] = [ x[3], x[1], x[2]]
    @inbounds xsym[:,42] = [-x[3], x[1], x[2]]
    @inbounds xsym[:,43] = [ x[3],-x[1], x[2]]
    @inbounds xsym[:,44] = [ x[3], x[1],-x[2]]
    @inbounds xsym[:,45] = [-x[3],-x[1], x[2]]
    @inbounds xsym[:,46] = [-x[3], x[1],-x[2]]
    @inbounds xsym[:,47] = [ x[3],-x[1],-x[2]]
    @inbounds xsym[:,48] = [-x[3],-x[1],-x[2]]
    return xsym
end