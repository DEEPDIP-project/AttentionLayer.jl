using Test
using Lux
using CUDA
using LuxCUDA
using AttentionLayer: attention, attentioncnn
using ComponentArrays: ComponentArray
using Random
using Zygote: Zygote

# Shared test parameters - reduced for faster execution
T = Float32
N = 32  # Reduced from 128 for faster testing
D = 2
batch = 2  # Reduced from 5 for faster testing
rng = Xoshiro(123)

# Helper function to test model functionality
function test_model_functionality(closure, θ, st, input_tensor, device_type)
    @testset "Model setup" begin
        @test closure !== nothing
        @test θ !== nothing
        @test st !== nothing
        @test typeof(closure) <: Lux.Chain
    end

    # Test model forward pass
    output = closure(input_tensor, θ, st)

    @testset "Model output" begin
        @test output !== nothing
        @test length(output) == 2  # Check that the output is a tuple
        @test isa(output[1], device_type)
        @test size(output[1]) == size(input_tensor)  # Check output size matches input
    end

    @testset "Automatic Differentiation" begin
        # Test differentiability with gradient computation
        grad = Zygote.gradient(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test !isnothing(grad)
        @test sum(grad) != 0.0  # Ensure gradients are non-zero

        # Test pullback mechanism
        y, back = Zygote.pullback(θ -> sum(abs2, closure(input_tensor, θ, st)[1]), θ)
        @test y ≈ sum(abs2, closure(input_tensor, θ, st)[1])
        y_bar = one(T)
        θ_bar = back(y_bar)
        @test θ_bar !== nothing
        @test sum(θ_bar) != 0.0
    end
end

# Test configurations covering different attention patterns
test_configs = [
    (
        name = "Small model with full attention",
        params = (
            N = 16,
            r = [2, 2],
            c = [4, 2],
            σ = [tanh, identity],
            b = [true, false],
            use_attention = [true, true],
            sum_attention = [false, false],
            patch_sizes = [8, 5],
            emb_sizes = [8, 8],
            n_heads = [2, 2],
        ),
    ),
    (
        name = "Medium model with first layer attention only",
        params = (
            N = N,
            r = [2, 2, 2],
            c = [8, 8, 2],
            σ = [tanh, tanh, identity],
            b = [true, true, false],
            use_attention = [true, false, false],
            sum_attention = [false, false, false],
            patch_sizes = [22, 20, 18],
            emb_sizes = [8, 8, 8],
            n_heads = [2, 2, 2],
        ),
    ),
    (
        name = "Medium model with sparse attention",
        params = (
            N = N,
            r = [2, 2, 2],
            c = [8, 8, 2],
            σ = [tanh, tanh, identity],
            b = [true, true, false],
            use_attention = [false, true, true],
            sum_attention = [false, false, false],
            patch_sizes = [22, 20, 18],
            emb_sizes = [8, 8, 8],
            n_heads = [2, 2, 2],
        ),
    ),
]

@testset "AttentionCNN Comprehensive Tests" begin

    for config in test_configs
        @testset "$(config.name)" begin
            params = config.params
            Ns = reverse([params.N + 2 * sum(params.r[1:i]) for i = 1:length(params.r)])

            @testset "CPU Implementation" begin
                # Create the model
                closure, θ, st = attentioncnn(
                    T = T,
                    D = D,
                    data_ch = D,
                    radii = params.r,
                    channels = params.c,
                    activations = params.σ,
                    use_bias = params.b,
                    use_attention = params.use_attention,
                    emb_sizes = params.emb_sizes,
                    Ns = Ns,
                    patch_sizes = params.patch_sizes,
                    n_heads = params.n_heads,
                    sum_attention = params.sum_attention,
                    rng = rng,
                    use_cuda = false,
                )

                # Test with CPU tensors
                input_tensor = rand(T, params.N, params.N, D, batch)
                test_model_functionality(closure, θ, st, input_tensor, Array)
            end

            @testset "GPU Implementation" begin
                if !CUDA.functional()
                    @test_skip "CUDA not available - skipping GPU tests"
                else
                    # Create the model for GPU
                    closure, θ, st = attentioncnn(
                        T = T,
                        D = D,
                        data_ch = D,
                        radii = params.r,
                        channels = params.c,
                        activations = params.σ,
                        use_bias = params.b,
                        use_attention = params.use_attention,
                        emb_sizes = params.emb_sizes,
                        Ns = Ns,
                        patch_sizes = params.patch_sizes,
                        n_heads = params.n_heads,
                        sum_attention = params.sum_attention,
                        rng = rng,
                        use_cuda = true,
                    )

                    # Test with GPU tensors
                    input_tensor = CUDA.rand(T, params.N, params.N, D, batch)
                    test_model_functionality(closure, θ, st, input_tensor, CuArray)
                end
            end
        end
    end

end
