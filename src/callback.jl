export thunk

"""
    ThunkIntegrand(f, x)

Store `f` and `x` to evaluate `f(x)` at a later time.
Can be employed in iterated integration
"""
struct ThunkIntegrand{T,d,X}
    f::T
    x::SVector{d,X}
end

(f::ThunkIntegrand)(x) = f.f(vcat(x, f.x))

"""
    thunk(f, x)

Delay the computation of f(x). Needed to normally evaluate an integrand in
nested integrals as employed by callbacks.
"""
thunk(f, x) = ThunkIntegrand(f, SVector(x))
thunk(f::ThunkIntegrand, x) = ThunkIntegrand(f.f, vcat(x, f.x))