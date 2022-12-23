export fermi, fermi′, fermi_window, fermi_window_limits

fermi(ω, β, μ) = fermi(ω-μ, β)
fermi(ω, β) = fermi(β*ω)
function fermi(x)
    y = exp(x)
    inv(one(y) + y)
end

fermi′(ω, β, μ) = fermi′(ω-μ, β)
fermi′(ω, β) = β*fermi′(β*ω)
function fermi′(x)
    y = cosh(x)
    -0.5inv(one(y)+y)
end

"Evaluates a unitless window function determined by the Fermi distribution"
fermi_window(ω, Ω, β, μ) = fermi_window(ω-μ, Ω, β)
fermi_window(ω, Ω, β) = fermi_window(β*ω, β*Ω)
fermi_window(x, y) = ifelse(y == zero(y), -fermi′(x), fermi_window_(x, y))

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
    fermi_window_limits(Ω, β [; atol=0.0, rtol=1e-20, μ=0.0])

These limits are designed for integrating over the cubic FBZ first, then over ω
restricted to the interval where the Fermi window is larger than `atol`.
Choosing `atol` wisely is important to integrating the entire region of
interest, so i
"""
function fermi_window_limits(Ω, β; atol=0.0, rtol=1e-20, μ=0.0)
    Δω = fermi_window_halfwidth(Ω, β, select_fermi_atol(β*Ω, atol, rtol))
    CubicLimits(SVector(μ-Ω/2-Δω), SVector(μ-Ω/2+Δω))
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