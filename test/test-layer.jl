using Test
using Lux
using AttentionLayer: attention
using ComponentArrays: ComponentArray
using Random
using Zygote: Zygote

@testset "Attention Layer" begin

    # Define parameters for the model
    T = Float32
    D = 2
    N = 16       # Define the spatial dimension for the attention layer input
    emb_size = 8
    patch_size = 4
    n_heads = 2
    rng = Xoshiro(123)

    # Test CNN Layer Setup
    r = [3]
    c = [4, 2]
    σ = [tanh]
    b = [false]
    CnnLayers = (
        (
            Lux.Conv(
                ntuple(α -> 2r[i] + 1, D),
                c[i] => c[i+1],
                σ[i];
                use_bias = b[i],
                pad = (ntuple(α -> 2r[i] + 1, D) .- 1) .÷ 2,
            ) for i in eachindex(r)
        )...,
    )

    # Verify CNN layers' setup
    @test CnnLayers != nothing

    # Define the Transformer model layers
    attention_layer = attention(N, D, emb_size, patch_size, n_heads; T = T)
    @test attention_layer != nothing

    # Combine Attention Layer with CNN in a Lux chain
    layers = (
        Lux.SkipConnection(
            attention_layer,
            (x, y) -> cat(x, y; dims = 3);
            name = "Attention",
        ),
        CnnLayers,
    )
    closure = Lux.Chain(layers...)
    θ, st = Lux.setup(rng, closure)
    θ = ComponentArray(θ)

    # Test model structure
    @test typeof(closure) <: Lux.Chain
    @test length(closure.layers) == 2  # Confirm both layers (Attention + CNN) are in chain

    # Define input tensor and pass through model
    input_tensor = rand(T, N, N, D, 1)  # Example input with shape (N, N, D, batch_size)
    output = closure(input_tensor, θ, st)

    # Test output dimensions after passing through the model
    expected_channels = D  # From the CNN output
    @test size(output[1]) == (N, N, expected_channels, 1)  # Check final output size

    # Test Differentiability by calculating gradients
    grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
    @test !isnothing(grad)  # Ensure gradients were successfully computed

end
