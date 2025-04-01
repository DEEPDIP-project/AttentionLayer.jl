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

# Test CNN Layer Setup
r = [3]
c = [4, 2]
σ = [tanh]
b = [false]

@testset "Attention Layer (CPU)" begin

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
    
    @testset "CNN setup" begin
        @test CnnLayers != nothing
    end

    # Define the Transformer model layers
    attention_layer = attention(N, D, emb_size, patch_size, n_heads; T = T)

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

    @testset "Attention Layer setup" begin
        @test attention_layer != nothing
        @test typeof(closure) <: Lux.Chain
        @test length(closure.layers) == 2  # Confirm both layers (Attention + CNN) are in chain
    end

    # Define input tensor and pass through model
    input_tensor = rand(T, N, N, D, batch)  # Example input with shape (N, N, D, batch_size)
    output = closure(input_tensor, θ, st)
    expected_channels = D  # From the CNN output

    @testset "Output dimensions" begin
        @test size(output[1]) == (N, N, expected_channels, batch)  # Check output size
        @test typeof(output[1][1]) == T  # Check output type
    end

    return
    @testset "AD" begin
        # Test Differentiability by calculating gradients
        grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test !isnothing(grad)  # Ensure gradients were successfully computed

        y, back = Zygote.pullback(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test y == sum(abs2, closure(input_tensor, θ, st)[1])
    end

end

@testset "Attention Layer (GPU)" begin
    return
    if !CUDA.functional()
        @info "CUDA not available"
        return
    end

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
    

    # Define the Transformer model layers
    attention_layer = attention(N, D, emb_size, patch_size, n_heads; T = T)

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
    dev = Lux.gpu_device()
    θ = θ |> dev
    st = st |> dev

    @testset "Attention Layer setup" begin
        @test attention_layer != nothing
        @test typeof(closure) <: Lux.Chain
        @test length(closure.layers) == 2  # Confirm both layers (Attention + CNN) are in chain
        @warn typeof(closure) 
        @warn typeof(θ) 
    end

    # Define input tensor and pass through model
    input_tensor = CUDA.rand(T, N, N, D, 1)  # Example input with shape (N, N, D, batch_size)
    output = closure(input_tensor, θ, st)
    expected_channels = D  # From the CNN output
    return

    @testset "Output dimensions" begin
        @test size(output[1]) == (N, N, expected_channels, 1)  # Check output size
        @test isa(output, CuArray)
    end

    @testset "AD" begin
        return
        # Test Differentiability by calculating gradients
        grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test !isnothing(grad)  # Ensure gradients were successfully computed

        y, back = Zygote.pullback(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test y == sum(abs2, closure(input_tensor, θ, st)[1])
    end

end