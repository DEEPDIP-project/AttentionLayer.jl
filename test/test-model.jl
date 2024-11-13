using Test
using Lux
using AttentionLayer: attention
using ComponentArrays: ComponentArray
using CoupledNODE: cnn
using Random
using Zygote: Zygote

@testset "Transformer Model" begin

    # Define parameters for the model
    T = Float32
    D = 2
    N = 32       # Define the spatial dimension for the attention layer input
    emb_size = 16
    patch_size = 4
    n_heads = 2
    d = emb_size ÷ n_heads  # Dimension per attention head
    rng = Xoshiro(123)

    # Test CNN Layer Setup
    CnnLayers, _, _ = cnn(;
        T = T,
        D = D,
        data_ch = 2 * D,                     # Input channels for CNN after concatenation
        radii = [3, 3],
        channels = [2, 2],
        activations = [tanh, identity],
        use_bias = [false, false],
        rng,
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
