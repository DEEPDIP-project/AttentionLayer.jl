using Test
using Lux
using CUDA
using LuxCUDA
using AttentionLayer: attention
using ComponentArrays: ComponentArray
using Random
using Zygote: Zygote

# Define parameters for the model
T = Float32
D = 2
N = 16       # Define the spatial dimension for the attention layer input
emb_size = 8
patch_size = 8
n_heads = 2
batch = 5
rng = Xoshiro(123)

# Define the attention layer
attention_layer = attention(N, D, emb_size, patch_size, n_heads; T = T)
closure = Lux.Chain(attention_layer)
θ, st = Lux.setup(rng, closure)
θ = ComponentArray(θ)

@testset "Attention Layer (CPU)" begin
    # Define input tensor and pass through model
    input_tensor = rand(T, N, N, D, batch)  # Example input with shape (N, N, D, batch_size)
    output = closure(input_tensor, θ, st)

    @testset "Output dimensions" begin
        @test size(output[1]) == (N, N, D, batch)  # Check output size
        @test typeof(output[1][1]) == T  # Check output type
        @test output[2] == st
        @test output[1] != input_tensor
    end

    @testset "AD" begin
        # Test Differentiability by calculating gradients
        grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test !isnothing(grad)  # Ensure gradients were successfully computed
        @test sum(grad) != 0.0  # Ensure gradients are not zero

        y, back = Zygote.pullback(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test y == sum(abs2, closure(input_tensor, θ, st)[1])
        y_bar = ones(T, size(y))
        θ_bar = back(y_bar)
        @test θ_bar != nothing
        @test sum(θ_bar) != 0.0  # Ensure gradients are not zero
    end

end

@testset "Attention Layer (GPU)" begin

    if !CUDA.functional()
        @error "CUDA is not available. Skipping GPU tests."
        return
    end
    # Move model and input to GPU
    closure = Lux.Chain(attention_layer)
    θ, st = Lux.setup(rng, closure)
    θ = ComponentArray(θ)
    dev = Lux.gpu_device()
    θ = θ |> dev
    st = st |> dev

    # Define input tensor and pass through model
    input_tensor = CUDA.rand(T, N, N, D, batch)  # Example input with shape (N, N, D, batch_size)
    output = closure(input_tensor, θ, st)

    @testset "Output dimensions" begin
        @test size(output[1]) == (N, N, D, batch)  # Check output size
        @test output[2] == st
        @test output[1] != input_tensor
        @test isa(output[1], CuArray)  # Check if output is on GPU
    end

    @testset "AD" begin
        # Test Differentiability by calculating gradients
        grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test !isnothing(grad)  # Ensure gradients were successfully computed
        @test sum(grad) != 0.0  # Ensure gradients are not zero

        y, back = Zygote.pullback(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test y == sum(abs2, closure(input_tensor, θ, st)[1])
        y_bar = CUDA.ones(T, size(y))
        θ_bar = back(y_bar)
        @test θ_bar != nothing
        @test sum(θ_bar) != 0.0  # Ensure gradients are not zero
    end

end
