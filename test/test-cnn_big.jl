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
N = 128
D = 2
batch = 5
rng = Xoshiro(123)
r = [2, 2, 2, 2, 2]
c = [24, 24, 24, 24, 2]
σ = [tanh, tanh, tanh, tanh, identity]
b = [true, true, true, true, false]
use_attention = [true, true, true, true, true]
sum_attention = [false, false, false, false, false]
Ns = reverse([N + 2 * sum(r[1:i]) for i = 1:length(r)])
patch_sizes = [37, 36, 35, 34, 33]
emb_sizes = [8, 8, 8, 8, 8]
n_heads = [2, 2, 2, 2, 2]

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
    if !CUDA.functional()
        @testset "CUDA not available" begin
            @test true
        end
        return
    end

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
