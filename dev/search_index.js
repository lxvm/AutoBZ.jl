var documenterSearchIndex = {"docs":
[{"location":"pages/app/interfaces/#Interfaces","page":"Interfaces","title":"Interfaces","text":"","category":"section"},{"location":"pages/app/interfaces/#Wannier90","page":"Interfaces","title":"Wannier90","text":"","category":"section"},{"location":"pages/app/interfaces/","page":"Interfaces","title":"Interfaces","text":"AutoBZ.Applications.parse_hamiltonian\nAutoBZ.Applications.load_hamiltonian","category":"page"},{"location":"pages/app/interfaces/#AutoBZ.Applications.parse_hamiltonian","page":"Interfaces","title":"AutoBZ.Applications.parse_hamiltonian","text":"parse_hamiltonian(filename)\n\nParse an ab-initio Hamiltonian output from Wannier90 into filename, extracting the fields (date_time, num_wann, nrpts, degen, irvec, C)\n\n\n\n\n\n","category":"function"},{"location":"pages/app/interfaces/#AutoBZ.Applications.load_hamiltonian","page":"Interfaces","title":"AutoBZ.Applications.load_hamiltonian","text":"load_hamiltonian(filename; period=1.0)\n\nLoad an ab-initio Hamiltonian output from Wannier90 into filename as an evaluatable FourierSeries whose periodicity can be set by the keyword argument period which defaults to setting the period along each dimension to 1.0. To define different periods for different dimensions, pass an SVector as the period.\n\n\n\n\n\n","category":"function"},{"location":"pages/app/interfaces/#Python","page":"Interfaces","title":"Python","text":"","category":"section"},{"location":"pages/app/interfaces/","page":"Interfaces","title":"Interfaces","text":"Julia code, including AutoBZ.jl, can be called from Python using the package PyJulia","category":"page"},{"location":"pages/app/interfaces/#Setup","page":"Interfaces","title":"Setup","text":"","category":"section"},{"location":"pages/app/interfaces/","page":"Interfaces","title":"Interfaces","text":"TL;DR","category":"page"},{"location":"pages/app/interfaces/","page":"Interfaces","title":"Interfaces","text":"$ julia -e 'import Pkg; Pkg.add(\"PyCall\")'\n$ python3 -m pip install julia","category":"page"},{"location":"pages/app/interfaces/","page":"Interfaces","title":"Interfaces","text":"If you want to, you can install PyJulia in a Python venv, but on the Julia side PyCall.jl must be installed in the default environment","category":"page"},{"location":"pages/app/interfaces/#Demo","page":"Interfaces","title":"Demo","text":"","category":"section"},{"location":"pages/app/interfaces/","page":"Interfaces","title":"Interfaces","text":"The Python snippet below shows how from the demos folder of the AutoBZ repository you can run one of the demo scripts:","category":"page"},{"location":"pages/app/interfaces/","page":"Interfaces","title":"Interfaces","text":"from julia.api import Julia\njl = Julia(compiled_modules=False)\n\n# julia environment setup in working directory 'demos'\njl.eval(\"\"\"\nimport Pkg\nPkg.activate(\".\")\nPkg.instantiate()\n\"\"\")\n\n# capture output of script\nout = jl.eval('include(\"DOS_test.jl\")')","category":"page"},{"location":"pages/app/interfaces/","page":"Interfaces","title":"Interfaces","text":"The first two lines are adapted for loading PyJulia on Debian systems.","category":"page"},{"location":"pages/app/integrands/#Integrands","page":"Integrands","title":"Integrands","text":"","category":"section"},{"location":"pages/app/integrands/#Functions","page":"Integrands","title":"Functions","text":"","category":"section"},{"location":"pages/app/integrands/","page":"Integrands","title":"Integrands","text":"AutoBZ.Applications.greens_function\nAutoBZ.Applications.spectral_function\nAutoBZ.Applications.dos_integrand\nAutoBZ.Applications.gamma_integrand\nAutoBZ.Applications.oc_integrand\nAutoBZ.Applications.fermi\nAutoBZ.Applications.fermi′\nAutoBZ.Applications.fermi_window\nAutoBZ.Applications.cosh_ratio\nAutoBZ.Applications.EXP_P1_SMALL_X","category":"page"},{"location":"pages/app/integrands/#AutoBZ.Applications.fermi_window","page":"Integrands","title":"AutoBZ.Applications.fermi_window","text":"Evaluates a unitless window function determined by the Fermi distribution\n\n\n\n\n\n","category":"function"},{"location":"pages/app/integrands/#Types","page":"Integrands","title":"Types","text":"","category":"section"},{"location":"pages/app/integrands/","page":"Integrands","title":"Integrands","text":"AutoBZ.Applications.GreensFunction\nAutoBZ.Applications.SpectralFunction\nAutoBZ.Applications.DOSIntegrand\nAutoBZ.Applications.GammaIntegrand\nAutoBZ.Applications.OCIntegrand\nAutoBZ.Applications.EquispaceOCIntegrand\nAutoBZ.Applications.AutoEquispaceOCIntegrand","category":"page"},{"location":"pages/app/integrands/#AutoBZ.Applications.GreensFunction","page":"Integrands","title":"AutoBZ.Applications.GreensFunction","text":"GreensFunction(H,ω,Σ,μ)\nGreensFunction(H,ω,Σ)\nGreensFunction(H,M)\n\nA struct that calculates the lattice Green's function from a Hamiltonian.\n\nG(kHomegaSigmamu) = ((omega + mu) I - H(k) - Sigma(omega))^-1\n\n\n\n\n\n","category":"type"},{"location":"pages/app/integrands/#AutoBZ.Applications.SpectralFunction","page":"Integrands","title":"AutoBZ.Applications.SpectralFunction","text":"SpectralFunction(::GreensFunction)\n\nA struct that calculates the imaginary part of the Green's function.\n\nA(kHωημ) = pi^-1 ImG(kHωημ)\n\n\n\n\n\n","category":"type"},{"location":"pages/app/integrands/#AutoBZ.Applications.DOSIntegrand","page":"Integrands","title":"AutoBZ.Applications.DOSIntegrand","text":"DOSIntegrand(::SpectralFunction)\n\nA struct whose integral gives the density of states.\n\nD(kHωημ) = operatornameTrA(kHωημ)\n\n\n\n\n\n","category":"type"},{"location":"pages/app/integrands/#AutoBZ.Applications.GammaIntegrand","page":"Integrands","title":"AutoBZ.Applications.GammaIntegrand","text":"GammaIntegrand(H, Σ, ω, Ω, μ)\nGammaIntegrand(H, ν₁, ν₂, ν₃, Mω, MΩ)\n\nA function whose integral over the BZ gives the transport distribution.\n\nGamma_alphabeta(k) = operatornameTrnu^alpha(k) A(komega) nu^beta(k) A(k omega+Omega)\n\n\n\n\n\n","category":"type"},{"location":"pages/app/integrands/#AutoBZ.Applications.OCIntegrand","page":"Integrands","title":"AutoBZ.Applications.OCIntegrand","text":"OCIntegrand(H, ν₁, ν₂, ν₃, Ω, β, η, μ)\n\nA function whose integral over the BZ and the frequency axis gives the optical conductivity\n\n\n\n\n\n","category":"type"},{"location":"pages/man/equispace_integration/#Equispace-integration","page":"Equispace integration","title":"Equispace integration","text":"","category":"section"},{"location":"pages/man/equispace_integration/#Routines","page":"Equispace integration","title":"Routines","text":"","category":"section"},{"location":"pages/man/equispace_integration/","page":"Equispace integration","title":"Equispace integration","text":"equispace_integration\nautomatic_equispace_integration","category":"page"},{"location":"pages/man/equispace_integration/#AutoBZ.equispace_integration","page":"Equispace integration","title":"AutoBZ.equispace_integration","text":"equispace_integration(f, l, npt; pre=nothing, pre_eval=generic_pre_eval, int_eval=generic_int_eval)\n\nEvaluate the integral of f over domain l using an equispace grid of npt points per dimension, optionally using precomputation pre\n\n\n\n\n\n","category":"function"},{"location":"pages/man/equispace_integration/#AutoBZ.automatic_equispace_integration","page":"Equispace integration","title":"AutoBZ.automatic_equispace_integration","text":"automatic_equispace_integration(f, a, b; kwargs)\nautomatic_equispace_integration(f, l::IntegrationLimits; npt1=0, pre1=0, npt2=0, pre2=0, pre_eval=generic_pre_eval, int_eval=generic_int_eval, atol=0.0, rtol=1e-3, npt_update=generic_npt_update, maxevals=typemax(Int64))\n\nAutomatically evaluates the integral of f over domain l to within the requested error tolerances atol and rtol. Allows optional precomputed data at two levels of grid refinement npt1, pre1 and npt2, pre2 as well as  customizable precomputation with pre_eval and evaluation/summation int_eval. Moreover, a function defining an update strategy for npt can be passed as npt_update.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/equispace_integration/#Customization","page":"Equispace integration","title":"Customization","text":"","category":"section"},{"location":"pages/man/equispace_integration/","page":"Equispace integration","title":"Equispace integration","text":"AutoBZ.generic_npt_update\nAutoBZ.generic_pre_eval\nAutoBZ.generic_int_eval","category":"page"},{"location":"pages/man/equispace_integration/#AutoBZ.generic_npt_update","page":"Equispace integration","title":"AutoBZ.generic_npt_update","text":"generic_npt_update(npt::Integer, f, atol, rtol)\n\nReturns a new npt to try and get another digit of accuracy from PTR. This fallback option is a heuristic, since the scaling of the error is generally problem-dependent.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/equispace_integration/#AutoBZ.generic_pre_eval","page":"Equispace integration","title":"AutoBZ.generic_pre_eval","text":"generic_pre_eval(f, l, npt)\n\nPrecomputes the grid points and weights to use for equispace quadrature of f on the domain l while applying the relevant symmetries to l to reduce the number of evaluation points.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/equispace_integration/#AutoBZ.generic_int_eval","page":"Equispace integration","title":"AutoBZ.generic_int_eval","text":"generic_int_eval(f, pre, dvol)\n\nSums the values of f on the precomputed grid points with corresponding integer weights and also multiplies by the mesh cell volume.\n\n\n\n\n\n","category":"function"},{"location":"pages/app/fourier/#Fourier-series","page":"Fourier series","title":"Fourier series","text":"","category":"section"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"Wannier-interpolated Hamiltonians are represented by Fourier series with a modest number of Fourier coefficients. The goal of this page of documentation is to describe the features, interface, and conventions of Fourier series evaluation as implemented by this library.","category":"page"},{"location":"pages/app/fourier/#Interface","page":"Fourier series","title":"Interface","text":"","category":"section"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"AutoBZ.Applications.AbstractFourierSeries\nAutoBZ.Applications.period\nAutoBZ.Applications.contract\nAutoBZ.Applications.value","category":"page"},{"location":"pages/app/fourier/#AutoBZ.Applications.AbstractFourierSeries","page":"Fourier series","title":"AutoBZ.Applications.AbstractFourierSeries","text":"AbstractFourierSeries{N}\n\nA supertype for Fourier series that are periodic maps R^N to V where V is any vector space. Typically these can be represented by N-dimensional arrays whose elements belong to the vector space. See the manual section on the AbstractFourierSeries interface.\n\n\n\n\n\n","category":"type"},{"location":"pages/app/fourier/#AutoBZ.Applications.period","page":"Fourier series","title":"AutoBZ.Applications.period","text":"period(f::AbstractFourierSeries{N}) where {N}\n\nReturn a NTuple{N} whose m-th element corresponds to the period of f along its m-th input dimension. Typically, these values set the units of length for the problem.\n\n\n\n\n\n","category":"function"},{"location":"pages/app/fourier/#AutoBZ.Applications.contract","page":"Fourier series","title":"AutoBZ.Applications.contract","text":"contract(f::AbstractFourierSeries{N}, x::Number, [dim=N]) where {N}\n\nReturn another Fourier series of dimension N-1 by summing over dimension dim of f with the phase factors evaluated at x. If N=1, this function should return an AbstractFourierSeries{0} that stores the evaluated Fourier series, but has no more input dimensions to contract.\n\nThe default of dim=N is motivated by preserving memory locality in Julia's column-major array format.\n\ncontract(f::AbstractFourierSeries{N}, x::SVector{M}) where {N,M}\n\nContract the outermost indices M of f in order of last(x) to first(x). If M>N, the default behavior is just to try and contract M indices, which will likely lead to an error.\n\n\n\n\n\n","category":"function"},{"location":"pages/app/fourier/#AutoBZ.Applications.value","page":"Fourier series","title":"AutoBZ.Applications.value","text":"value(::AbstractFourierSeries{0})\n\nReturn the evaluated Fourier series whose indices have all been contracted. Typically, this value has the same units as the Fourier series coefficients.\n\n\n\n\n\n","category":"function"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"Additonally, concrete subtypes of AbstractFourierSeries must have an element type, which they can do by extending Base.eltype with a method. For example, if a type MyFourierSeries <: AbstractFourierSeries always returns ComplexF64 outputs, then the correct eltype method to define would be:","category":"page"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"Base.eltype(::Type{MyFourierSeries}) = ComplexF64","category":"page"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"The type returned should correspond to the vector space V of the output space of the Fourier series, i.e. the output of value should be of this type. For good performance, the eltype should be a concrete type and should be inferrable.","category":"page"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"With the above implemented, several methods which define functors for AbstractFourierSeries allow the user (and integration routines) to evaluate the type like a function with the f(x) syntax.","category":"page"},{"location":"pages/app/fourier/#Types","page":"Fourier series","title":"Types","text":"","category":"section"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"The concrete types listed below all implement the AbstractFourierSeries interface and should cover most use cases.","category":"page"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"AutoBZ.Applications.FourierSeries\nAutoBZ.Applications.FourierSeriesDerivative\nAutoBZ.Applications.OffsetFourierSeries\nAutoBZ.Applications.ManyFourierSeries\nAutoBZ.Applications.ManyOffsetsFourierSeries\nAutoBZ.Applications.BandEnergyVelocity","category":"page"},{"location":"pages/app/fourier/#AutoBZ.Applications.FourierSeries","page":"Fourier series","title":"AutoBZ.Applications.FourierSeries","text":"FourierSeries(coeffs, period::SVector{N,Float64}) where {N}\n\nConstruct a Fourier series whose coefficients are given by the coefficient array array coeffs whose eltype should support addition and scalar multiplication, and whose periodicity on the ith axis is given by period[i]. This type represents the Fourier series\n\nf(vecx) = sum_vecn in mathcal I C_vecn exp(i2piveck_vecncdotoverrightarrowx)\n\nwhere i = sqrt-1 is the imaginary unit, C is the array coeffs, mathcal I is CartesianIndices(C), vecn is a CartesianIndex and veck_vecn is equal to n_jp_j in the jth position with p_j the jth element of period. Because of the choice to use Cartesian indices to set the phase factors, typically the indices of coeffs should be specified by using an OffsetArray.\n\n\n\n\n\n","category":"type"},{"location":"pages/app/fourier/#AutoBZ.Applications.FourierSeriesDerivative","page":"Fourier series","title":"AutoBZ.Applications.FourierSeriesDerivative","text":"FourierSeriesDerivative(f::FourierSeries{N}, a::SVector{N}) where {N}\n\nRepresent the differential of Fourier series f by a multi-index a of derivatives, e.g. [1,2,...], whose ith entry represents the order of differentiation on the ith input dimension of f. Mathematically, this means\n\nleft( prod_j=1^N partial_x_j^a_j right) f(vecx) = sum_vecn in mathcal I left( prod_j=1^N (i 2pi k_j)^a_j right) C_vecn exp(i2piveck_vecncdotoverrightarrowx)\n\nwhere partial_x_j^a_j represents the a_jth derivative of x_j, i = sqrt-1 is the imaginary unit, C is the array coeffs, mathcal I is CartesianIndices(C), vecn is a CartesianIndex and veck_vecn is equal to n_jp_j in the jth position with p_j the jth element of period. Because of the choice to use Cartesian indices to set the phase factors, typically the indices of coeffs should be specified by using an OffsetArray. Also, note that termwise differentiation of the Fourier series results in additional factors of i2pi which should be anticipated for the use case. Also, note that this type can be used to represent fractional differentiation or integration by suitably choosing the a_js.\n\nThis is a 'lazy' representation of the derivative because instead of differentiating by computing all of the Fourier coefficients of the derivative upon constructing the object, the evaluator waits until it contracts the differentiated dimension to evaluate the new coefficients.\n\n\n\n\n\n","category":"type"},{"location":"pages/app/fourier/#AutoBZ.Applications.OffsetFourierSeries","page":"Fourier series","title":"AutoBZ.Applications.OffsetFourierSeries","text":"OffsetFourierSeries(f::AbstractFourierSeries{N}, q::SVector{N,Float64}) where {N}\n\nRepresent a Fourier series whose argument is offset by the vector vecq and evaluates it as f(vecx-vecq).\n\n\n\n\n\n","category":"type"},{"location":"pages/app/fourier/#AutoBZ.Applications.ManyFourierSeries","page":"Fourier series","title":"AutoBZ.Applications.ManyFourierSeries","text":"ManyFourierSeries(fs::AbstractFourierSeries{N}...) where {N}\n\nRepresents a tuple of Fourier series of the same dimension and periodicity and contracts them all simultaneously.\n\n\n\n\n\n","category":"type"},{"location":"pages/app/fourier/#AutoBZ.Applications.ManyOffsetsFourierSeries","page":"Fourier series","title":"AutoBZ.Applications.ManyOffsetsFourierSeries","text":"ManyOffsetsFourierSeries(f, qs..., [origin=true])\n\nRepresent a Fourier series evaluated at many different points, and contract them all simultaneously, returning them in the order the qs were passed, i.e. (f(x-qs[1]), f(x-qs[2]), ...) The origin keyword decides whether or not to evaluate f without an offset, and if origin is true, the value of f evaluated without an offset will be returned in the first position of the output.\n\n\n\n\n\n","category":"type"},{"location":"pages/app/fourier/#AutoBZ.Applications.BandEnergyVelocity","page":"Fourier series","title":"AutoBZ.Applications.BandEnergyVelocity","text":"BandEnergyVelocity(H::FourierSeries{3})\n\nThis constructor takes a Fourier series representing the Hamiltonian and also evaluates the band velocities so that the return value after all the dimensions are contracted is a tuple containing (H, ν₁, ν₂, ν₃). The band velocities are defined by dipole operators nu_alpha = -fracihbar partial_k_alpha H where k_alpha is one of three input dimensions of H and hbar=1. Note that differentiation changes the units to have an additional dimension of length, so if H has units of dimensions of energy, nu has dimensions of energy times length. The caller is responsible for transforming the units of the velocity (i.e. hbar) if they want other units, which can usually be done as a post-processing step.\n\n\n\n\n\n","category":"function"},{"location":"pages/app/fourier/#Methods","page":"Fourier series","title":"Methods","text":"","category":"section"},{"location":"pages/app/fourier/","page":"Fourier series","title":"Fourier series","text":"AutoBZ.Applications.contract(::AutoBZ.Applications.AbstractFourierSeries)","category":"page"},{"location":"pages/app/fourier/#AutoBZ.Applications.contract-Tuple{AutoBZ.Applications.AbstractFourierSeries}","page":"Fourier series","title":"AutoBZ.Applications.contract","text":"contract(f::FourierSeries{N}, x::Number, [dim=N]) where {N}\n\nContract index dim of the coefficients of f at the spatial point x. The default dim is the outermost dimension to preserve memory locality.\n\n\n\n\n\ncontract(f::FourierSeriesDerivative{N}, x::Number, [dim=N]) where {N}\n\nContract index dim of the coefficients of f at the spatial point x. The default dim is the outermost dimension to preserve memory locality.\n\n\n\n\n\n","category":"method"},{"location":"pages/man/adaptive_integration/#Adaptive-integration","page":"Adaptive integration","title":"Adaptive integration","text":"","category":"section"},{"location":"pages/man/adaptive_integration/#Routines","page":"Adaptive integration","title":"Routines","text":"","category":"section"},{"location":"pages/man/adaptive_integration/","page":"Adaptive integration","title":"Adaptive integration","text":"tree_integration\niterated_integration","category":"page"},{"location":"pages/man/adaptive_integration/#AutoBZ.tree_integration","page":"Adaptive integration","title":"AutoBZ.tree_integration","text":"tree_integration(f, a, b)\ntree_integration(f, ::CubicLimits)\n\nCalls HCubature to perform multi-dimensional integration of f over a cube.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/adaptive_integration/#AutoBZ.iterated_integration","page":"Adaptive integration","title":"AutoBZ.iterated_integration","text":"iterated_integration(f, ::IntegrationLimits)\n\nCalls HCubature to perform iterated 1D integration of f over a domain parametrized by IntegrationLimits. Accepts a callback function whose arguments are f and the evaluation point, x, as a keyword argument. The callback can return a modified integrand to the next inner integral, but the default is thunk which delays the computation to the innermost integral.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/adaptive_integration/#Customization","page":"Adaptive integration","title":"Customization","text":"","category":"section"},{"location":"pages/man/adaptive_integration/","page":"Adaptive integration","title":"Adaptive integration","text":"thunk\nAutoBZ.ThunkIntegrand\nAutoBZ.default_error_callback","category":"page"},{"location":"pages/man/adaptive_integration/#AutoBZ.thunk","page":"Adaptive integration","title":"AutoBZ.thunk","text":"thunk(f, x)\n\nDelay the computation of f(x). Needed to normally evaluate an integrand in nested integrals as employed by callbacks.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/adaptive_integration/#AutoBZ.ThunkIntegrand","page":"Adaptive integration","title":"AutoBZ.ThunkIntegrand","text":"ThunkIntegrand(f, x)\n\nStore f and x to evaluate f(x) at a later time. Can be employed in iterated integration\n\n\n\n\n\n","category":"type"},{"location":"pages/man/adaptive_integration/#AutoBZ.default_error_callback","page":"Adaptive integration","title":"AutoBZ.default_error_callback","text":"Choose a new set of error tolerances for the next inner integral.\n\n\n\n\n\n","category":"function"},{"location":"pages/demo/#Demos","page":"Demos","title":"Demos","text":"","category":"section"},{"location":"pages/demo/#DOS-of-the-integer-lattice-tight-binding-model","page":"Demos","title":"DOS of the integer lattice tight-binding model","text":"","category":"section"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"To demonstrate setting up a DOS calculation with AutoBZ, we consider a tight-binding model on the n-dimensional integer lattice with lattice constant a and hopping strength t0:","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"H = -t sum_i in Z^n sum_j=1^n ketibrai+hatj + keti+hatjbrai","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"Solving this model by employing Bloch's theorem yields the following band","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"H(k_1 ldots k_n) = -t(cos(k_1 a) + cdots + cos(k_n a))","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"We shall input this Hamiltonian by constructing the equivalent Fourier series","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"using StaticArrays\nusing OffsetArrays\n\nusing AutoBZ\nusing AutoBZ.Applications\n\nn = 3 # arbitrary positive integer\na = fill(1.0, SVector{n})\nax = repeat([-1:1], n)\nC = zeros(SMatrix{1,1,ComplexF64,1}, ntuple(_ -> 3, n))\nfor i in 1:n, j in (-1, 1)\n    C[CartesianIndex(ntuple(k -> k == i ? 2+j : 2, n))] = SMatrix{1,1,ComplexF64,1}(0.5)\nend\nH = FourierSeries(OffsetArray(C, ax...), a)","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"Then we can define the integration problem to compute DOS, defined by the integral","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"operatornameDOS(omega) = int_textBZ dveck operatornameTrImomega+mu-H(veck)+ieta","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"where mu is the chemical potential and eta is a constant scattering rate.","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"ω = 1.0*n # frequency\nη = 0.1 # broadening\nμ = 0.0 # chemical potential\nΣ = EtaEnergy(η) # self energy\nD = DOSIntegrand(H, ω, Σ, μ) # integrand\n\n# construct IBZ integration limits\nc = CubicLimits(H.period)\nt = TetrahedralLimits(c)\n\n# set error tolerances\natol = 1e-3\nrtol = 0.0\n\niterated_integration(D, t; callback=contract, atol=atol, rtol=rtol)","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"You will find a working example of this model in the DOS_example.jl demo that computes DOS over a range of frequencies for this model","category":"page"},{"location":"pages/demo/#Custom-adaptive-integrand","page":"Demos","title":"Custom adaptive integrand","text":"","category":"section"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"For integrands that can be evaluated by Wannier interpolation, the following data are necessary to define an integrand:","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"the integrand evaluator\na Fourier series\nadditional parameters","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"Consider implementing custom integrands using this generic template with a few associated methods","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"struct WannierIntegrand{TF,TS<:AbstractFourierSeries,TP}\n    f::TF\n    s::TS\n    p::TP\nend\ncontract(w::WannierIntegrand, x) = WannierIntegrand(w.f, contract(w.s, x), p)\n(w::WannierIntegrand)(x::SVector{1}) = w(only(x))\n(w::WannierIntegrand)(x::Number) = w.f(w.s(x), w.p...)","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"This integrand will be compatible with adaptive integration routines like iterated_integration.","category":"page"},{"location":"pages/demo/#Tight-binding","page":"Demos","title":"Tight binding","text":"","category":"section"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"For example, we can replicate the preceding tight-binding example by defining an integrand with the custom integrand type","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"using LinearAlgebra\ndos(H_k::AbstractMatrix, ω, μ, η) = -tr(imag(inv(complex(ω+μ, η)*I-H_k)))/pi\nD = WannierIntegrad(dos, H, (ω, μ, η))","category":"page"},{"location":"pages/demo/#Graphene-example-with-ManyOffsetsFourierSeries","page":"Demos","title":"Graphene example with ManyOffsetsFourierSeries","text":"","category":"section"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"Let's study an example motivated by graphene whose Hamiltonian is given by a tight-binding model on the hexagonal lattice with lattice constant a and hopping amplitude t. Applying Bloch's theorem to each triangular sublattice brings the Hamiltonian into block-diagonal form, where each block is of the form","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"-t\nbeginpmatrix\n0  f(k)\n f^*(k)  0\nendpmatrix","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"where f(k) = e^ikcdotdelta_1 + e^ikcdotdelta_2 + e^ikcdotdelta_3 and delta_1 = ahatx delta_2 = a(-12hatx+sqrt32haty) delta_3 = a(-12hatx-sqrt32haty). To exactly construct this Fourier series, we will have to rotate basis so that these vectors are precisely integer linear combinations of the new lattice vectors. Note that by defining hata_1 = (delta_1 - delta_3)3a = (hatx + 1sqrt3haty)2 hata_2 = (delta_1 - delta_2)3a = (hatx - 1sqrt3haty)2 we can write delta_1 = a(hata_1 + hata_2) delta_2 = a(hata_1 - 2hata_2) delta_3 = a(-2hata_1 + hata_2). Therefore our coordinate transformation matrix, T^-1 from Cartesian coordinates to the triangular lattice is","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"T = frac12\nbeginpmatrix\n1  1sqrt3\n 1  -1sqrt3\nendpmatrix\nqquad\nT^-1 =\nbeginpmatrix\n1  1\n sqrt3  -sqrt3\nendpmatrix","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"and note operatornamedet(T) = 12sqrt3. Now the corresponding reciprocal lattice vectors are constructed by the relation hatb_i = epsilon_ij (hatz times hata_j) and rescaling so that hatb_i cdot hata_j = 2pidelta_ij. This yields hatb_1 = 2pi(hatx+sqrt3haty) = 4pi(hata_1 - 2hata_2) hatb_2 = 2pi(hatx-sqrt3haty) = 4pi(2hata_1 - hata_2). We would now interpret k in this basis, and need to observe that if a is the lattice constant of the hexagonal lattice, then sqrt3a is the lattice constant of the triangular lattice, and 2pisqrt3a is the lattice constant of the reciprocal lattice. However, we will have to rescale by factors of operatornamedetT because of our coordinate transformations.","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"Having chosen this suitable basis for k and x, we can now express the k-dependence of the block Hamiltonian as","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"f(k) = e^ikcdotdelta_1 + e^ikcdotdelta_2 + e^ikcdotdelta_3\n= e^iakcdot(hata_1 + hata_2) + e^iakcdot(hata_1 - 2hata_2) + e^iakcdot(-2hata_1 + hata_2)","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"which is amenable to a Fourier series representation.","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"Suppose that the integral we want to calculate is","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"g(vecq) = int_textBZ dk_x dk_y fraclambda(xi(veck)) - lambda(xi(veck-vecq))xi(veck) - xi(veck-vecq)","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"where xi(veck) = operatornamedet(H(veck)) and lambda(omega) = partial_T f(omega) is the temperature derivative of the Fermi distribution. Since the integrand requires evaluation of the Hamiltonian at various k-points simultaneously, the ManyOffsetsFourierSeries type can be used to do this. Moreover, AutoBZ.Applications has functions to evaluate Fermi functions and their derivatives. Putting everything together leads us to the code example below","category":"page"},{"location":"pages/demo/","page":"Demos","title":"Demos","text":"using StaticArrays\nusing OffsetArrays\n\nusing AutoBZ\nusing AutoBZ.Applications\n\na = 1.0\nC = zeros(SMatrix{2,2,ComplexF64,4}, (5,5))\nC[1,1]   = SMatrix{2,2,ComplexF64,4}(0,0,1,0)\nC[-1,-1] = SMatrix{2,2,ComplexF64,4}(0,1,0,0)\nC[1,-2]  = SMatrix{2,2,ComplexF64,4}(0,0,1,0)\nC[-1,2]  = SMatrix{2,2,ComplexF64,4}(0,1,0,0)\nC[-2,1]  = SMatrix{2,2,ComplexF64,4}(0,0,1,0)\nC[2,-1]  = SMatrix{2,2,ComplexF64,4}(0,1,0,0)\nH = FourierSeries(OffsetArray(C, -2:2, -2:2), a)\n\nT = 1.0 # K\nkB = 8.617333262e-5 # eV/K\nq = SVector{3,Float64}(1.2, 2.8, 8.1) # arbitrary\nf = ManyOffsetsFourierSeries(H, q)\n\nlambda(x, T, kB) = -AutoBZ.Applications.fermi′(inv(kB*T), x)/(kB*T^2)\nintegrand_(f, T, kB) = (lambda(det(f[1]), T, kB) - lambda(det(f[2]), T, kB))/(det(f[1])-det(f[2]))\nintegrand = WannierIntegrand(integrand_, f, (T, kB))\n\nc = CubicLimits(H.period)\n\n# set error tolerances\natol = 1e-3\nrtol = 0.0\n\niterated_integration(D, c; callback=contract, atol=atol, rtol=rtol)","category":"page"},{"location":"pages/demo/#Custom-equispace-integrand","page":"Demos","title":"Custom equispace integrand","text":"","category":"section"},{"location":"pages/demo/#Fixed-BZ-grid","page":"Demos","title":"Fixed BZ grid","text":"","category":"section"},{"location":"pages/demo/#Automatic-BZ-grid","page":"Demos","title":"Automatic BZ grid","text":"","category":"section"},{"location":"pages/man/integration_limits/#Integration-limits","page":"Integration limits","title":"Integration limits","text":"","category":"section"},{"location":"pages/man/integration_limits/#Interface","page":"Integration limits","title":"Interface","text":"","category":"section"},{"location":"pages/man/integration_limits/","page":"Integration limits","title":"Integration limits","text":"lower\nupper\nbox\nsymmetries\nndims\nnsyms","category":"page"},{"location":"pages/man/integration_limits/#AutoBZ.lower","page":"Integration limits","title":"AutoBZ.lower","text":"lower(::IntegrationLimits)\n\nReturn the lower limit of the next variable of integration. If a vector is returned, then the integration routine may attempt to multidimensional integration.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/integration_limits/#AutoBZ.upper","page":"Integration limits","title":"AutoBZ.upper","text":"upper(::IntegrationLimits)\n\nReturn the upper limit of the next variable of integration. If a vector is returned, then the integration routine may attempt to multidimensional integration.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/integration_limits/#AutoBZ.box","page":"Integration limits","title":"AutoBZ.box","text":"box(::IntegrationLimits)\n\nReturn an iterator of tuples that for each dimension returns a tuple with the lower and upper limits of the integration domain without symmetries applied.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/integration_limits/#AutoBZ.symmetries","page":"Integration limits","title":"AutoBZ.symmetries","text":"symmetries(::IntegrationLimits)\n\nReturn an iterator over the symmetry transformations that the parametrization has used to reduce the volume of the integration domain.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/integration_limits/#Base.ndims","page":"Integration limits","title":"Base.ndims","text":"ndims(::IntegrationLimits{d})\n\nReturns d. This is a type-based rule.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/integration_limits/#AutoBZ.nsyms","page":"Integration limits","title":"AutoBZ.nsyms","text":"nsyms(::IntegrationLimits)\n\nReturn the number of symmetries that the parametrization has used to reduce the volume of the integration domain.\n\n\n\n\n\n","category":"function"},{"location":"pages/man/integration_limits/#Types","page":"Integration limits","title":"Types","text":"","category":"section"},{"location":"pages/man/integration_limits/","page":"Integration limits","title":"Integration limits","text":"IntegrationLimits\nCubicLimits\nCompositeLimits","category":"page"},{"location":"pages/man/integration_limits/#AutoBZ.IntegrationLimits","page":"Integration limits","title":"AutoBZ.IntegrationLimits","text":"IntegrationLimits{d}\n\nRepresents a set of integration limits over d variables. Realizations of this type should implement lower and upper, which return the lower and upper limits of integration along some dimension, rescale which represents the number of symmetries of the BZ which are used by the realization to reduce the BZ (the integrand over the limits gets multiplied by this factor), and a functor that accepts a single numeric argument and returns another realization of that type (in order to do nested integration). Thus the realization is also in control of the order of variables of integration and must coordinate this behavior with their integrand. Instances should also be static structs.\n\n\n\n\n\n","category":"type"},{"location":"pages/man/integration_limits/#AutoBZ.CubicLimits","page":"Integration limits","title":"AutoBZ.CubicLimits","text":"CubicLimits(a, [b])\n\nStore integration limit information for a hypercube with vertices a and b. If b is not passed as an argument, then the lower limit defaults to zero(a).\n\n\n\n\n\n","category":"type"},{"location":"pages/man/integration_limits/#AutoBZ.CompositeLimits","page":"Integration limits","title":"AutoBZ.CompositeLimits","text":"CompositeLimits(::Tuple{Vararg{IntegrationLimits}})\n\nConstruct a collection of limits which yields the first limit followed by the second, and so on.\n\n\n\n\n\n","category":"type"},{"location":"pages/man/integration_limits/#Routines","page":"Integration limits","title":"Routines","text":"","category":"section"},{"location":"pages/man/integration_limits/","page":"Integration limits","title":"Integration limits","text":"vol\nsymmetrize\nAutoBZ.discretize_equispace","category":"page"},{"location":"pages/man/integration_limits/#AutoBZ.vol","page":"Integration limits","title":"AutoBZ.vol","text":"vol(::IntegrationLimits)\n\nReturn the volume of the full domain without the symmetries applied\n\n\n\n\n\n","category":"function"},{"location":"pages/man/integration_limits/#AutoBZ.symmetrize","page":"Integration limits","title":"AutoBZ.symmetrize","text":"symmetrize(::IntegrationLimits, x)\nsymmetrize(::IntegrationLimits, xs...)\n\nTransform x by the symmetries of the parametrization used to reduce the domain, thus mapping the value of x on the parametrization to the full domain. When the integrand is a scalar, this is equal to nsyms(l)*x. When the integrand is a vector, this is sum(S*x for S in symmetries(l)). When the integrand is a matrix, this is sum(S*x*S' for S in symmetries(l)).\n\n\n\n\n\n","category":"function"},{"location":"pages/man/integration_limits/#AutoBZ.discretize_equispace","page":"Integration limits","title":"AutoBZ.discretize_equispace","text":"discretize_equispace(::IntegrationLimits, ::Integer)\n\nReturn an iterator of 2-tuples containing integration nodes and weights that correspond to an equispace integration grid with the symmetry transformations applied to it.\n\n\n\n\n\n","category":"function"},{"location":"#AutoBZ.jl-documentation","page":"Home","title":"AutoBZ.jl documentation","text":"","category":"section"}]
}
