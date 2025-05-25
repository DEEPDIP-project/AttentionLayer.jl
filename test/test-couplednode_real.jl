using Test
using Adapt
using Lux
using JLD2
using AttentionLayer: attention, attentioncnn
using ComponentArrays: ComponentArray
using Optimisers: Adam, ClipGrad, OptimiserChain
using Random
using Zygote: Zygote
using CUDA
using LuxCUDA
using CoupledNODE
using IncompressibleNavierStokes
using NeuralClosure
using OrdinaryDiffEqTsit5

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
    nles = conf["params"]["nles"][1]
    T = Float32

    closure, θ_start, st = NS.load_model(conf)
    device = x -> adapt(Array, x)

    @info "CNN warm up run"
    u = randn(Float32, nles + 2, nles + 2, 2, 10) |> device
    θ = θ_start |> device
    output, _ = closure(u, θ, st)

    @test size(output) == (nles + 2, nles + 2, 2, 10)
    @test isa(output, Array)

    # get params
    params = NS.load_params(conf)
    device(x) = adapt(params.backend, x)

    # Get the setup in the format expected by the CoupledNODE
    function getsetup(; params, nles)
        Setup(;
            x = ntuple(α -> range(params.lims..., nles + 1), params.D),
            params.Re,
            params.backend,
            params.bodyforce,
            params.issteadybodyforce,
        )
    end
    setup = getsetup(; params, nles)
    psolver = default_psolver(setup)
    setup = []
    for nl in nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nl + 1), params.D)
        push!(setup, Setup(; x = x, Re = params.Re, params.backend))
    end

    # Load data
    data_train = load("data_train.jld2", "data_test")


    # Create the io array
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    io_train = NS.create_io_arrays_posteriori(data_train, setup[1], device)

    # Create the dataloader
    θ = device(copy(θ_start))
    nunroll = 5
    nunroll_valid = 5
    dataloader_post = NS.create_dataloader_posteriori(
        io_train;
        nunroll = nunroll,
        rng = Random.Xoshiro(24),
        device = device,
    )

    # Create the right hand side and the loss
    dudt_nn = NS.create_right_hand_side_with_closure(setup[1], psolver, closure, st)
    griddims = ((:) for _ = 1:D)
    loss = CoupledNODE.create_loss_post_lux(dudt_nn, griddims, griddims;)
    callbackstate = trainstate = nothing


    # For testing reason, explicitely set up the probelm
    # Notice that this is automatically done in CoupledNODE
    u, t = dataloader_post()
    x = u[griddims..., :, 1, 1]
    y = u[griddims..., :, 1, 2:end] # remember to discard sol at the initial time step
    tspan, dt, prob, pred = nothing, nothing, nothing, nothing # initialize variable outside allowscalar do.
    function get_tspan(t)
        return (Array(t)[1], Array(t)[end])
    end
    tspan = get_tspan(t)
    prob = ODEProblem(dudt_nn, x, tspan, θ)
    pred = Array(solve(prob, Tsit5(); u0 = x, p = θ, adaptive = true, saveat = Array(t)))

    # Test the forward pass
    @test size(pred[:, :, :, 2:end]) == size(y)


    # Test the backward pass
    p = prob.p
    y = prob.u0
    f = prob.f
    λ = CUDA.zero(prob.u0)
    _dy, back = Zygote.pullback(y, p) do u, p
        vec(f(u, p, t))
    end
    tmp1, tmp2 = back(λ)
    @test size(tmp1) == (nles+2, nles+2, 2)
    @test size(tmp2) == (184144,)
    @test isa(tmp1, CuArray)  # Check if tmp1 is on GPU

    # Final integration test of the entire train interface
    l, trainstate = CoupledNODE.train(
        closure,
        θ,
        st,
        dataloader_post,
        loss;
        tstate = trainstate,
        nepochs = 2,
        alg = OptimiserChain(Adam(T(1.0e-3)), ClipGrad(0.1)),
        cpu = true,
    )
    @test isnan(l) == false
    @test trainstate.step == 2
    @test any(isnan, trainstate.parameters) == false

end
