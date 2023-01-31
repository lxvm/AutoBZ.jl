export fermi, fermi′, fermi_window, fermi_window_limits

"""
    fermi(x)

Evaluates a Fermi distribution with unitless input
```math
f(x) = \\frac{1}{e^{x}+1}
```
"""
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
    -0.5inv(one(y)+y)
end

"""
    fermi_window(x, y)

Evaluates a unitless window function with unitless inputs determined by the
Fermi distribution ``f`` and defined by
```math
\\chi(x, y) = \\frac{f(x) - f(x+y)}{y}
```
In the case `y==0` then this simplifies to the derivative of the Fermi distribution.
"""
fermi_window(x, y) = y == zero(y) ? -fermi′(x) : fermi_window_(x, y)

fermi_window_(x, y) = fermi_window_(promote(float(x), float(y))...)
function fermi_window_(x::T, y::T) where {T<:AbstractFloat}
    half_y = y*T(0.5)
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
function fermi_window_limits(Ω, β; atol=0.0, rtol=1e-20)
    Δω = fermi_window_halfwidth(Ω, β, select_fermi_atol(β*Ω, atol, rtol))
    (-Ω/2-Δω, -Ω/2+Δω)
end
select_fermi_atol(x, atol, rtol) = ifelse(x == zero(x), max(atol, 0.25rtol), max(atol, tanh(x/4)/x*rtol))
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
    if β == zero(β) || atol == zero(atol)
        Inf
    elseif 1/4 > atol
        inv(β)*log(1/atol - 2)
    else
        error("choose `atol` under 1/4, the maximum of the Fermi window")
    end
end

fermi_window_halfwidth_(x, atol) = fermi_window_halfwidth_(float(x), atol)
function fermi_window_halfwidth_(x::AbstractFloat, atol)
    y = x/2
    log(2cosh(y)*(tanh(y)/(x*atol) - 1))
end
function fermi_window_halfwidth_(x::T, atol) where {T<:Union{Float32,Float64}}
    y = x/2
    abs_y = abs(y)
    y_large = Base.Math.H_LARGE_X(T)-1.0 # subtract 1 so 2cosh(x) won't overflow
    ifelse(abs_y > y_large, abs_y, log(2cosh(y))) + log(tanh(y)/(x*atol) - 1)
    # to be exact, add log1p(exp(-2abs_y)) to abs_y, but this is lost to roundoff
end
fermi_window_halfwidth_(x::Float16, atol) = Float16(fermi_window_halfwidth_(Float32(x), atol))