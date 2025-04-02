using Test
using Lux
using CUDA
using LuxCUDA
using AttentionLayer: attention, attentioncnn
using ComponentArrays: ComponentArray
using Random
using Zygote: Zygote

# Define parameters for the model
T = Float32
N = 16
D = 2
batch = 5
rng = Xoshiro(123)
r = [2, 2]
c = [4, 2]
σ = [tanh, identity]
b = [true, false]
emb_sizes = [8, 8]
Ns = reverse([N + 2 * sum(r[1:i]) for i = 1:length(r)])
patch_sizes = [8, 5]
n_heads = [2, 2]
use_attention = [true, true]
sum_attention = [false, false]

@testset "AttentionCNN (CPU)" begin

    # Create the model
    closure, θ, st = attentioncnn(
        T = T,
        D = D,
        data_ch = D,
        radii = r,
        channels = c,
        activations = σ,
        use_bias = b,
        use_attention = use_attention,
        emb_sizes = emb_sizes,
        Ns = Ns,
        patch_sizes = patch_sizes,
        n_heads = n_heads,
        sum_attention = sum_attention,
        rng = rng,
        use_cuda = false,
    )

    @testset "Model setup" begin
        @test closure != nothing
        @test θ != nothing
        @test st != nothing
        # Test model structure
        @test typeof(closure) <: Lux.Chain
        # Confirm that we have all the layers: collocate + padder + attention + cnn + decollocate
        @test length(closure.layers) == 1 + D + 2 + 2 + 1
    end

    # Define input tensor and pass through model
    input_tensor = rand(T, N, N, D, batch)  # Example input with shape (N, N, D, batch_size)
    output = closure(input_tensor, θ, st)

    @testset "Model output" begin
        @test output != nothing
        @test length(output) == 2  # Check that the output is a tuple
        @test isa(output[1], Array)
        @test size(output[1]) == (N, N, D, batch)  # Check final output size
    end

    @testset "AD" begin
        # Test Differentiability by calculating gradients
        grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test !isnothing(grad)  # Ensure gradients were successfully computed
        @test sum(grad) != 0.0  # Ensure gradients are not zero

        y, back = Zygote.pullback(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test y ≈ sum(abs2, closure(input_tensor, θ, st)[1])
        y_bar = ones(T, size(y))
        θ_bar = back(y_bar)
        @test θ_bar != nothing
        @test sum(θ_bar) != 0.0  # Ensure gradients are not zero
    end

end

@testset "AttentionCNN (GPU)" begin

    # Create the model
    closure, θ, st = attentioncnn(
        T = T,
        D = D,
        data_ch = D,
        radii = r,
        channels = c,
        activations = σ,
        use_bias = b,
        use_attention = use_attention,
        emb_sizes = emb_sizes,
        Ns = Ns,
        patch_sizes = patch_sizes,
        n_heads = n_heads,
        sum_attention = sum_attention,
        rng = rng,
        use_cuda = true,
    )

    @testset "Model setup" begin
        @test closure != nothing
        @test θ != nothing
        @test st != nothing
        # Test model structure
        @test typeof(closure) <: Lux.Chain
        # Confirm that we have all the layers: collocate + padder + attention + cnn + decollocate
        @test length(closure.layers) == 1 + D + 2 + 2 + 1
    end

    # Define input tensor and pass through model
    input_tensor = CUDA.rand(T, N, N, D, batch)  # Example input with shape (N, N, D, batch_size)
    output = closure(input_tensor, θ, st)

    @testset "Model output" begin
        @test output != nothing
        @test length(output) == 2  # Check that the output is a tuple
        @test isa(output[1], CuArray)
        @test size(output[1]) == (N, N, D, batch)  # Check final output size
    end

    @testset "AD" begin
        # Test Differentiability by calculating gradients
        grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test !isnothing(grad)  # Ensure gradients were successfully computed
        @test sum(grad) != 0.0  # Ensure gradients are not zero

        y, back = Zygote.pullback(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test y ≈ sum(abs2, closure(input_tensor, θ, st)[1])
        y_bar = CUDA.ones(T, size(y))
        θ_bar = back(y_bar)
        @test θ_bar != nothing
        @test sum(θ_bar) != 0.0  # Ensure gradients are not zero
    end

end

@testset "Input sizes" begin
    # We have to test that the model can handle different input sizes
    # because data a_priori have no padding, while a_posteriori have

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
    Ns = reverse([N + 2 * sum(r[1:i]) for i = 1:length(r)])
    patch_sizes = [8, 5]
    n_heads = [2, 2]
    use_attention = [true, true]
    sum_attention = [false, false]

    # Create the model
    closure, θ, st = attentioncnn(
        T = T,
        D = D,
        data_ch = D,
        radii = r,
        channels = c,
        activations = σ,
        use_bias = b,
        use_attention = use_attention,
        emb_sizes = emb_sizes,
        Ns = Ns,
        patch_sizes = patch_sizes,
        n_heads = n_heads,
        sum_attention = sum_attention,
        rng = rng,
        use_cuda = false,
    )

    @info "There are $(length(θ)) parameters"


    # Define input tensor and pass through model
    input_tensor = rand(T, N, N, D, 1)
    output = closure(input_tensor, θ, st)
    expected_channels = D  # From the CNN output
    @test size(output[1]) == (N, N, expected_channels, 1)  # Check final output size

    pad_input_tensor = rand(T, N + 2, N + 2, D, 1)
    pad_input_tensor[2:end-1, 2:end-1, :, :] = input_tensor
    pad_output = closure(pad_input_tensor, θ, st)
    @test size(pad_output[1]) == (N + 2, N + 2, expected_channels, 1)

    pad_input_tensor = rand(T, N + 10, N + 10, D, 1)
    pad_input_tensor[6:end-5, 6:end-5, :, :] = input_tensor
    pad_output = closure(pad_input_tensor, θ, st)
    @test size(pad_output[1]) == (N + 10, N + 10, expected_channels, 1)

end
