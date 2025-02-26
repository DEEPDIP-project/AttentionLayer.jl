using Test
using Lux
using AttentionLayer: attention, attentioncnn
using ComponentArrays: ComponentArray
using Random
using Zygote: Zygote

@testset "CNN + attention" begin

    # Define parameters for the model
    T = Float32
    N = 16
    D = 2
    rng = Xoshiro(123)
    r = [2, 2]
    c = [4, 2]
    σ = [tanh, identity]
    b = [true, false]
    emb_sizes = [8, 8]
    patch_sizes = [8, 5]
    n_heads = [2, 2]
    use_attention = [true, true]
    sum_attention = [false, false]

    # Create the model
    closure, θ, st = attentioncnn(
        T = T,
        N = N,
        D = D,
        data_ch = D,
        radii = r,
        channels = c,
        activations = σ,
        use_bias = b,
        use_attention = use_attention,
        emb_sizes = emb_sizes,
        patch_sizes = patch_sizes,
        n_heads = n_heads,
        sum_attention = sum_attention,
        rng = rng,
        use_cuda = false,
    )

    @info "There are $(length(θ)) parameters"

    # Test model structure
    @test typeof(closure) <: Lux.Chain
    # Confirm that we have all the layers: collocate + padder + attention + cnn + decollocate
    @test length(closure.layers) == 1 + D + 2 + 2 + 1

    # Define input tensor and pass through model
    input_tensor = rand(T, N, N, D, 1)  # Example input with shape (N, N, D, batch_size)
    output = closure(input_tensor, θ, st)

    # Test output dimensions after passing through the model
    expected_channels = D  # From the CNN output
    @test size(output[1]) == (N, N, expected_channels, 1)  # Check final output size

    # Test Differentiability by calculating gradients
    grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
    @test !isnothing(grad)  # Ensure gradients were successfully computed

    y, back = Zygote.pullback(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
    @test y == sum(abs2, closure(input_tensor, θ, st)[1])

end
