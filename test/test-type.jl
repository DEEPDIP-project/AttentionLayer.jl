using Test
using Lux
using CUDA
using LuxCUDA
using AttentionLayer: attention, attentioncnn
using ComponentArrays: ComponentArray
using Random
using Zygote: Zygote

function get_test_params(T, N = 16)
    return (
        T = T,
        N = N,
        D = 2,
        batch = 2,
        rng = Xoshiro(123),
        r = [2, 2],
        c = [8, 2],
        σ = [tanh, identity],
        b = [true, false],
        use_attention = [true, false],
        sum_attention = [false, false],
        Ns = reverse([N + 2 * sum([2, 2][1:i]) for i = 1:2]),
        patch_sizes = [8, 5],
        emb_sizes = [4, 4],
        n_heads = [2, 2],
    )
end

@testset "AttentionCNN Type Tests" begin

    # Test both Float32 and Float64
    for T in [Float32, Float64]
        @testset "Type $T (CPU)" begin
            params = get_test_params(T)

            # Create the model
            closure, θ, st = attentioncnn(
                T = params.T,
                D = params.D,
                data_ch = params.D,
                radii = params.r,
                channels = params.c,
                activations = params.σ,
                use_bias = params.b,
                use_attention = params.use_attention,
                emb_sizes = params.emb_sizes,
                Ns = params.Ns,
                patch_sizes = params.patch_sizes,
                n_heads = params.n_heads,
                sum_attention = params.sum_attention,
                rng = params.rng,
                use_cuda = false,
            )

            @testset "Model setup and type consistency" begin
                @test closure !== nothing
                @test θ !== nothing
                @test st !== nothing
                @test typeof(closure) <: Lux.Chain

                # Check that parameters have the correct type
                @test eltype(θ) == T
            end

            # Define input tensor with correct type
            input_tensor = rand(T, params.N, params.N, params.D, params.batch)
            output = closure(input_tensor, θ, st)

            @testset "Model output and type preservation" begin
                @test output !== nothing
                @test length(output) == 2
                @test isa(output[1], Array)
                @test size(output[1]) == (params.N, params.N, params.D, params.batch)

                # Check that output has the correct element type
                @test eltype(output[1]) == T
            end

            @testset "Gradient computation" begin
                # Test basic gradient computation
                loss_fn = θ -> sum(abs2, closure(input_tensor, θ, st)[1])
                grad = Zygote.gradient(loss_fn, θ)
                @test grad !== nothing
                @test grad[1] !== nothing

                # Verify gradient types match parameter types
                @test eltype(grad[1]) == T
            end
        end

        # GPU tests
        @testset "Type $T (GPU)" begin
            if !CUDA.functional()
                @test_skip "CUDA not available"
                continue
            end

            params = get_test_params(T)

            # Create the model
            closure, θ, st = attentioncnn(
                T = params.T,
                D = params.D,
                data_ch = params.D,
                radii = params.r,
                channels = params.c,
                activations = params.σ,
                use_bias = params.b,
                use_attention = params.use_attention,
                emb_sizes = params.emb_sizes,
                Ns = params.Ns,
                patch_sizes = params.patch_sizes,
                n_heads = params.n_heads,
                sum_attention = params.sum_attention,
                rng = params.rng,
                use_cuda = true,
            )

            @testset "GPU model setup and type consistency" begin
                @test closure !== nothing
                @test θ !== nothing
                @test st !== nothing
                @test typeof(closure) <: Lux.Chain

                # Check that parameters have the correct type
                @test eltype(θ) == T
            end

            # Define input tensor with correct type on GPU
            input_tensor = CUDA.rand(T, params.N, params.N, params.D, params.batch)
            output = closure(input_tensor, θ, st)

            @testset "GPU model output and type preservation" begin
                @test output !== nothing
                @test length(output) == 2
                @test isa(output[1], CuArray)
                @test size(output[1]) == (params.N, params.N, params.D, params.batch)

                # Check that output has the correct element type
                @test eltype(output[1]) == T
            end

            @testset "GPU gradient computation" begin
                # Test basic gradient computation
                loss_fn = θ -> sum(abs2, closure(input_tensor, θ, st)[1])
                grad = Zygote.gradient(loss_fn, θ)
                @test grad !== nothing
                @test grad[1] !== nothing

                # Verify gradient types match parameter types
                @test eltype(grad[1]) == T
            end
        end
    end
end
