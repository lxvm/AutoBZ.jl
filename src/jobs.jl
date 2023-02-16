# parallelization

"""
    batch_smooth_param(xs, nthreads)

If the cost of a calculation smoothly varies with the parameters `xs`, then
batch `xs` into `nthreads` groups where the `i`th element of group `j` is
`xs[j+(i-1)*nthreads]`
"""
function batch_smooth_param(xs, nthreads)
    batches = [Tuple{Int,eltype(xs)}[] for _ in 1:min(nthreads, length(xs))]
    for (i, x) in enumerate(xs)
        push!(batches[mod(i-1, nthreads)+1], (i, x))
    end
    batches
end

"""
    parallel_integration(f::AbstractIntegrator, ps; nthreads=Threads.nthreads())

Evaluate the `AbstractIntegrator` `f` at each of the parameters `ps` in
parallel. Returns a named tuple `(I, E, t, p)` containing the integrals `I`, the
extra data from the integration routine `E`, timings `t`, and the original
parameters `p`. The parameter layout in `ps` should such that `f(ps[i]...)` runs
"""
function parallel_integration(f::AbstractIntegrator, ps; nthreads=Threads.nthreads())
    T = Base.promote_op(first∘f, eltype(ps))
    ints = Vector{T}(undef, length(ps))
    # extra = Vector{???}(undef, length(ps))
    ts = Vector{Float64}(undef, length(ps))
    @info "Beginning parameter sweep using $(routine(f))"
    @info "using $nthreads threads for parameter parallelization"
    batches = batch_smooth_param(ps, nthreads)
    t = time()
    Threads.@threads for batch in batches
        f_ = deepcopy(f) # to avoid data races for in place integrators
        for (i, p) in batch
            @info @sprintf "starting parameter %i" i
            t_ = time()
            # TODO: ints[i], extra[i] = quad_return(routine(f), f_(p...))
            ts[i] = time() - t_
            @info @sprintf "finished parameter %i in %e (s) wall clock time" i ts[i]
        end
    end
    @info @sprintf "Finished parameter sweep in %e (s) CPU time and %e (s) wall clock time" sum(ts) (time()-t)
    (I=ints, t=ts, p=ps)
end

# enables kpt parallelization by default for all BZ integrals
# with symmetries
function evalptr(rule, npt, f::FourierIntegrand, B::SMatrix{N,N}, syms, min_per_thread=1, nthreads=Threads.nthreads()) where N
    n = length(rule.x)
    acc = rule.w[n]*ptr_integrand(f, rule.x[n]) # unroll first term in sum to get right types
    n == 1 && return acc*det(B)/length(syms)/npt^N
    runthreads = min(nthreads, div(n-1, min_per_thread)) # choose the actual number of threads
    d, r = divrem(n-1, runthreads)
    partial_sums = fill!(Vector{typeof(acc)}(undef, runthreads), zero(acc)) # allocations :(
    Threads.@threads for i in Base.OneTo(runthreads)
        # batch nodes into `runthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        @inbounds for j in 1:jmax
            partial_sums[i] += rule.w[offset + j]*ptr_integrand(f, rule.x[offset + j])
        end
    end
    for part in partial_sums
        acc += part
    end
    acc*det(B)/length(syms)/npt^N
end
# without symmetries
function evalptr(rule, npt, f::FourierIntegrand, B::SMatrix{N,N}, ::Nothing, min_per_thread=1, nthreads=Threads.nthreads()) where N
    n = length(rule.x)
    acc = ptr_integrand(f, rule.x[n]) # unroll first term in sum to get right types
    n == 1 && return acc*det(B)/npt^N
    runthreads = min(nthreads, div(n-1, min_per_thread)) # choose the actual number of threads
    d, r = divrem(n-1, runthreads)
    partial_sums = fill!(Vector{typeof(acc)}(undef, runthreads), zero(acc)) # allocations :(
    Threads.@threads for i in Base.OneTo(runthreads)
        # batch nodes into `runthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        @inbounds for j in 1:jmax
            partial_sums[i] += ptr_integrand(f, rule.x[offset + j])
        end
    end
    for part in partial_sums
        acc += part
    end
    acc*det(B)/npt^N
end