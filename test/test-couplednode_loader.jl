using Test
using Adapt
using Lux
using JLD2
using AttentionLayer: attention, attentioncnn
using ComponentArrays: ComponentArray
using Optimisers: Adam, ClipGrad, OptimiserChain
using Random
using Zygote: gradient
using CUDA
using LuxCUDA
using CoupledNODE
using IncompressibleNavierStokes
using NeuralClosure
using OrdinaryDiffEqTsit5


@testset "CoupledNode loader (CPU)" begin
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    conf = NS.read_config("./config.yaml")
    conf["params"]["backend"] = CPU()

    closure, θ_start, st = NS.load_model(conf)
    device = x -> adapt(Array, x)

    @info "CNN warm up run"
    u = randn(Float32, 32 + 2, 32 + 2, 2, 10) |> device
    θ = θ_start |> device
    #u = rand(Float32, 32+2, 32+2, 2, 10)
    #θ = θ_start |> Lux.gpu_device()
    output, _ = closure(u, θ, st)

    @test size(output) == (32 + 2, 32 + 2, 2, 10)
    @test isa(output, Array)

    g = gradient(θ -> sum(closure(u, θ, st)[1]), θ)
    @test sum(g) != 0.0  # Ensure gradients are not zero

end

@testset "CoupledNode loader (GPU)" begin
    if !CUDA.functional()
        @testset "CUDA not available" begin
            @test true
        end
        return
    end
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    conf = NS.read_config("./config.yaml")
    conf["params"]["backend"] = CUDABackend()

    closure, θ_start, st = NS.load_model(conf)
    device = x -> adapt(CuArray, x)

    @info "CNN warm up run"
    u = randn(Float32, 32 + 2, 32 + 2, 2, 10) |> device
    θ = θ_start |> device
    #u = CUDA.rand(Float32, 32+2, 32+2, 2, 10)
    #θ = θ_start |> Lux.gpu_device()
    output, _ = closure(u, θ, st)

    @test size(output) == (32 + 2, 32 + 2, 2, 10)
    @test isa(output, CuArray)

    g = gradient(θ -> sum(closure(u, θ, st)[1]), θ)
    @test sum(g) != 0.0  # Ensure gradients are not zero

end
