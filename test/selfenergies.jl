using Test
using AutoBZ

let
se_scalar_uniform = join([
    "10",
    ("$x    0.0    -0.1" for x in range(-15, 15, length=10))...
], '\n')
se_scalar_nonuniform = join([
    "10",
    ("$(x+y)    0.0    -0.1" for (y, x) in zip(range(-1, 1, length=10), range(-15, 15, length=10)))...
], '\n')

se_diagonal_uniform = join([
    "10", "2",
    ("$x    1    0.0    -0.1\n$x    2    0.0    -0.2" for x in range(-15, 15, length=10))...
], '\n')
se_diagonal_nonuniform = join([
    "10", "2",
    ("$(x+y)    1    0.0    -0.1\n$(x+y)    2    0.0    -0.2" for (y, x) in zip(range(-1, 1, length=10), range(-15, 15, length=10)))...
], '\n')

se_matrix_uniform = join([
    "10", "2",
    ("$x    1    1    0.0    -0.1\n$x    1    2    0.0    -0.4\n$x    2    1    0.0    -0.5\n$x    2    2    0.0    -0.2" for x in range(-15, 15, length=10))...
], '\n')
se_matrix_nonuniform = join([
    "10", "2",
    ("$(x+y)    1    1    0.0    -0.1\n$(x+y)    1    2    0.0    -0.4\n$(x+y)    2    1    0.0    -0.5\n$(x+y)    2    2    0.0    -0.2" for (y, x) in zip(range(-1, 1, length=10), range(-15, 15, length=10)))...
], '\n')

d = 2
C = graphene()
bz = load_bz(CubicSymIBZ(d))
h_w = coeffs2FourierHamiltonian(C; gauge=Wannier())
h_h = coeffs2FourierHamiltonian(C; gauge=Hamiltonian())
hv_w = let h=coeffs2FourierHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), LAPACKEigen(); gauge=Wannier()); end
hv_h = let h=coeffs2HermitianHamiltonian(C); GradientVelocityInterp(h, bz.A, EigenProblem(h(rand(d))), JLEigen(); gauge=Hamiltonian()); end
μ = 0.1
abstol=1e-2
reltol=0.0
alg = PTR(; npt=50)
falg = QuadGKJL(order=4)

ω = 0.0

ω₁= 0.0
ω₂= 0.1

β = 10.0

Ω = 0.0

for (n, file_str) in enumerate([
    se_scalar_uniform,
    se_scalar_nonuniform,
    se_diagonal_uniform,
    se_diagonal_nonuniform,
    se_matrix_uniform,
    se_matrix_nonuniform,
])
    Σ = load_self_energy(IOBuffer(file_str))

    solver_h = TrGlocSolver(Σ, h_h, bz, alg; ω, μ, abstol, reltol)
    solver_w = TrGlocSolver(Σ, h_w, bz, alg; ω, μ, abstol, reltol)
    sol_h = solve!(solver_h)
    sol_w = solve!(solver_w)
    # test that the Hamiltonian and Wannier gauges agree
    @test sol_h.value ≈ sol_w.value atol=abstol rtol=reltol
    
    solver_h = TransportDistributionSolver(Σ, hv_h, bz, alg; ω₁, ω₂, μ, abstol, reltol)
    solver_w = TransportDistributionSolver(Σ, hv_w, bz, alg; ω₁, ω₂, μ, abstol, reltol)
    sol_h = solve!(solver_h)
    sol_w = solve!(solver_w)
    # test that the Hamiltonian and Wannier gauges agree
    @test sol_h.value ≈ sol_w.value atol=abstol rtol=reltol

    solver_h = ElectronDensitySolver(h_h, bz, alg, Σ, falg; β, μ, abstol, reltol)
    solver_w = ElectronDensitySolver(h_w, bz, alg, Σ, falg; β, μ, abstol, reltol)
    sol_h = solve!(solver_h)
    sol_w = solve!(solver_w)
    # test that the Hamiltonian and Wannier gauges agree
    @test sol_h.value ≈ sol_w.value atol=abstol rtol=reltol

    solver_h = OpticalConductivitySolver(hv_h, bz, alg, Σ, falg; β, μ, Ω, abstol, reltol)
    solver_w = OpticalConductivitySolver(hv_w, bz, alg, Σ, falg; β, μ, Ω, abstol, reltol)
    sol_h = solve!(solver_h)
    sol_w = solve!(solver_w)
    # test that the Hamiltonian and Wannier gauges agree
    @test sol_h.value ≈ sol_w.value atol=abstol rtol=reltol
end
end