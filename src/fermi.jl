"""
    fermi(β, ω)
    fermi(x)

Evaluates a Fermi distribution with unitless input
```math
f(x) = \\frac{1}{e^{x}+1}
```
"""
function fermi(β, ω)
    x = β*ω
    isfinite(β)     ? fermi(x)  :
    isnan(β)        ? x         : # in last case, β is inf
    ω <= zero(ω)    ? one(x)    : zero(x)   # special case ω=0, since otherwise NaN
end
function fermi(x)
    y = exp(x)
    inv(one(y) + y)
end

"""
    fermi′(x)

Evaluates a first derivative of the Fermi distribution with unitless input
```math
\\partial_{x} f(x) = -\\frac{1}{2(\\cosh(x)+1)}
```
Note that the analytic expression above can be rewritten many ways using
hypertrigonometric identities.
"""
function fermi′(x)
    y = cosh(x)
    -inv(one(y)+one(y)+y+y)
end

"""
    fermi_window(β, ω, Ω)
    fermi_window(x, y)

Evaluates a unitless window function with unitless inputs determined by the
Fermi distribution ``f`` and defined by
```math
\\chi(x, y) = \\frac{f(x) - f(x+y)}{y}
```
In the case `y==0` then this simplifies to the derivative of the Fermi distribution.

See also [`fermi`](@ref) and [`fermi′`](@ref).
"""
function fermi_window(β::T, ω::T, Ω::T) where {T<:AbstractFloat}
    if isinf(β)
        if iszero(ω) && iszero(Ω)
            β   # this is technically a Dirac delta function :(
        elseif -Ω <= ω <= zero(Ω)
            inv(Ω)
        else
            zero(β)
        end
    else
        β*fermi_window(β*ω, β*Ω)
    end
end
function fermi_window(x::T, y::T) where {T<:AbstractFloat}
    return iszero(y) ? -fermi′(x) : fermi_window_(x, y)
end
fermi_window(args...) = fermi_window(promote(map(float, args)...)...)

function fermi_window(β, ω, Ω)
    uβ = float(oneunit(β))
    return uβ*fermi_window(β/uβ, ω*uβ, Ω*uβ)
end
fermi_window_(x, y) = fermi_window_(promote(float(x), float(y))...)
function fermi_window_(x::T, y::T) where {T<:AbstractFloat}
    half_y = y/2
    (tanh(half_y)/y)/(one(T)+cosh_ratio(x+half_y, half_y))
end

cosh_ratio(x, y) = cosh(x)/cosh(y)
function cosh_ratio(x::T, y::T) where {T<:Union{Float32,Float64}}
    abs_x = abs(x)
    abs_y = abs(y)
    arg_large = Base.Math.H_LARGE_X(T)
    arg_small = EXP_P1_SMALL_X(T)
    if max(abs_x, abs_y) < arg_large
        cosh(x)/cosh(y)
    elseif arg_large <= abs_x && -2*abs_y > arg_small
        exp(abs_x-abs_y)/(one(T)+exp(-2*abs_y))
    elseif arg_large <= abs_y && -2*abs_x > arg_small
        exp(abs_x-abs_y)*(one(T)+exp(-2*abs_x))
    else
        exp(abs_x-abs_y)
    end
end

# log(eps(T))
EXP_P1_SMALL_X(::Type{Float64}) = -36.04365338911715
EXP_P1_SMALL_X(::Type{Float32}) = -15.942385f0

"""
    fermi_window_limits(Ω, β [; atol=0.0, rtol=1e-20])

Returns limits `(a,b)` over ω restricted to the interval where the Fermi window
is larger than `max(atol,rtol*fermi_window(0,β*Ω))`. Choosing `atol` and `rtol`
wisely is important to integrating the entire region of interest, since this is
a truncation of an infinite interval, and should be tested for convergence.
"""
function fermi_window_limits(Ω, β; atol=zero(Ω*β), rtol=iszero(atol) ? eps(one(atol)) : zero(atol))
    Δω = fermi_window_halfwidth(Ω, β, select_fermi_atol(β*Ω, atol, rtol))
    (-Ω/2-Δω, -Ω/2+Δω)
end
select_fermi_atol(x, atol, rtol) = ifelse(x == zero(x), max(atol, rtol/4), max(atol, tanh(x/4)/x*rtol))
"""
    fermi_window_halfwidth(Ω, β, atol)
    fermi_window_halfwidth(β, atol)

One can show that β*Ω*fermi_window(ω, β, Ω) =
-tanh(β*Ω/2)/(cosh(β*(ω+Ω/2))/cosh(β*Ω/2)+1) >
-tanh(β*Ω/2)/(exp(abs(β*(ω+Ω/2)))/2cosh(β*Ω/2)+1)
as well as when Ω==0, β*fermi_window(ω, β, 0.0) =
and these can be inverted to give a good bound on the width of the frequency
window for which the Fermi window function is greater than `atol`. Returns half
the width of this window.
"""
function fermi_window_halfwidth(Ω, β, atol)
    isinf(β) && return Ω/2
    x = β*Ω
    if x == zero(x) || atol == zero(atol)
        fermi_window_halfwidth(β, atol)
    elseif tanh(x/4)/x > atol
        inv(β)*fermi_window_halfwidth_(x, atol)
    else
        error("choose `atol` under tanh(β*Ω/4)/(β*Ω), the maximum of the Fermi window")
    end
end
function fermi_window_halfwidth(β, atol)
    t = inv(β)
    if β == zero(β) || atol == zero(atol)
        t
    elseif 1 > 4*atol
        t*log(inv(atol) - 2one(atol))
    else
        error("choose `atol` under 1/4, the maximum of the Fermi window")
    end
end

fermi_window_halfwidth_(x, atol) = fermi_window_halfwidth_(float(x), atol)
function fermi_window_halfwidth_(x::T, atol::T) where {T<:AbstractFloat}
    y = x/2
    log(2cosh(y)*(tanh(y)/(x*atol) - one(T)))
end
function fermi_window_halfwidth_(x::T, atol::T) where {T<:Union{Float32,Float64}}
    y = x/2
    abs_y = abs(y)
    y_large = Base.Math.H_LARGE_X(T)-one(T) # subtract 1 so 2cosh(x) won't overflow
    ifelse(abs_y > y_large, abs_y, log(2cosh(y))) + log(tanh(y)/(x*atol) - one(T))
    # to be exact, add log1p(exp(-2abs_y)) to abs_y, but this is lost to roundoff
end
fermi_window_halfwidth_(x::Float16, atol::Float16) = Float16(fermi_window_halfwidth_(Float32(x), Float32(atol)))

fermi_window_maximum(β, Ω) = iszero(Ω) ? oftype(1/Ω, β/4) : tanh(β*Ω/4)/Ω # == fermi_window(β, -Ω/2, Ω)

function fermi_function_limits(β; atol=zero(one(β)), rtol=iszero(atol) ? eps(one(atol)) : zero(atol))
    tol = max(atol, rtol) # since max of fermi function is always 1
    u = -log(tol)/β
    return (-typemax(typeof(u)), u)
end
