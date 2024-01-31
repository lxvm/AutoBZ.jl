using Test, AutoBZ, OffsetArrays

@testset "shift!" begin

    C = [
        0.0 0.1 0.0
        0.2 0.4 0.2
        0.3 0.5 0.3
        0.2 0.4 0.2
        0.0 0.1 0.0
    ]

    f1 = FourierSeries(Array(C), period=1.0, offset=(-3,-2))
    f2 = FourierSeries(OffsetMatrix(Array(C), -2:2, -1:1), period=1.0)

    x1 = (0.0, 0.0)
    x2 = (0.7, 0.3)
    fx1 = f1(x1)
    fx2 = f1(x2)
    @test f1(x1) ≈ f2(x1) ≈ fx1
    @test f1(x2) ≈ f2(x2) ≈ fx2
    
    for o in [-2.1, -0.2, 0.1, 1.3, 4.5]
        shift!(f1, o)
        shift!(f2, o)
        @test f1(x1) ≈ f2(x1) ≈ fx1-o
        @test f1(x2) ≈ f2(x2) ≈ fx2-o
        shift!(f1, -o)
        shift!(f2, -o)
    end

end

@testset "droptol" begin

    Cpad = [
        0.0 0.0 0.0 0.0
        0.0 0.1 0.0 0.0
        0.2 0.4 0.2 0.0
        0.3 0.5 0.3 0.0
        0.2 0.4 0.2 0.0
        0.0 0.1 0.0 0.0
    ]
    OCpad = OffsetMatrix(Array(Cpad), -3:2, -1:2)
    C = Cpad[begin+1:end, begin:end-1]
    OC = OffsetMatrix(C, -2:2, -1:1)

    @test @inferred(AutoBZ.droptol(Cpad,  CartesianIndex(4,2), eps())) == (C, CartesianIndex(3,2))
    @test @inferred(AutoBZ.droptol(OCpad, CartesianIndex(0,0), eps())) == (C, CartesianIndex(3,2))
    @test @inferred(AutoBZ.droptol(C,  CartesianIndex(3,2), eps())) == (C, CartesianIndex(3,2))
    @test @inferred(AutoBZ.droptol(OC, CartesianIndex(0,0), eps())) == (C, CartesianIndex(3,2))


    Cpad2 = [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.1 0.0 0.0
        0.2 0.4 0.2 0.0
        0.3 0.5 0.3 0.0
        0.2 0.4 0.2 0.0
        0.0 0.1 0.0 0.0
    ]
    OCpad2 = OffsetMatrix(Array(Cpad2), -4:2, -1:2)
    @test @inferred(AutoBZ.droptol(Cpad2,  CartesianIndex(5,2), eps())) == (C, CartesianIndex(3,2))
    @test @inferred(AutoBZ.droptol(OCpad2, CartesianIndex(0,0), eps())) == (C, CartesianIndex(3,2))
end