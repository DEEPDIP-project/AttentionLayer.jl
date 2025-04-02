using Test
using Lux
using CUDA
using LuxCUDA
using AttentionLayer: attention, attentioncnn
using ComponentArrays: ComponentArray
using Random
using Zygote: Zygote

# *** We have to test that the model can handle different input sizes
#     because data a_priori have no padding, while a_posteriori have

# Define parameters for the model
T = Float32
N = 40
D = 2
batch = 5
rng = Xoshiro(123)
r = [2, 2]
c = [24, 2]
σ = [tanh, identity]
b = [true, false]
emb_sizes = [124, 124]
Ns = [40, 36]
patch_sizes = [8, 6]
n_heads = [4, 4]
use_attention = [true, true]
sum_attention = [false, false]


@testset "Input sizes (CPU)" begin

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


    # Define input tensor and pass through model
    input_tensor = rand(T, N, N, D, batch)
    output = closure(input_tensor, θ, st)
    @test size(output[1]) == (N, N, D, batch)  # Check final output size


end


@testset "Input sizes (GPU)" begin
    if !CUDA.functional()
        @info "CUDA not available"
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


    # Define input tensor and pass through model
    input_tensor = CUDA.rand(T, N, N, D, batch)
    output = closure(input_tensor, θ, st)
    @test size(output[1]) == (N, N, D, batch)  # Check final output size
    
end
