@testset "utils/utils.jl" begin
    
    # softplus
    s = (3,4,2)
    x = GenerativeModels.softplus_safe.(randn(Float32, s))
    @test size(x) == s
    @test all(x .> 0)
    @test eltype(x) == Float32
    @test eltype(softplus_safe.(x, Float64)) == Float64

    # stack layers
    s = (8, 6)
    x = randn(Float32, s)
    zdim = 2
    lsize = (s[1], 4, zdim)
    m = GenerativeModels.stack_layers(lsize, relu)
    y = m(x)

    @test length(m.layers) == 2
    @test size(y) == (zdim, s[2])
    @test eltype(y) == Float32
    @test size(m.layers[1].W) == (lsize[2], lsize[1])

    m = GenerativeModels.stack_layers(lsize, relu, last = sigmoid)
    y = m(x)
    @test size(y) == (2, 6)

    m = GenerativeModels.layer_builder([3,3,2], "tanh", "linear")
    @test length(m.layers) == 2
    @test m.layers[1] isa Dense
    @test m.layers[1].σ == tanh
    @test m.layers[2] isa Dense
    @test m.layers[2].σ == identity

end
