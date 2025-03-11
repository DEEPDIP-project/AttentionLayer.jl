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
using CoupledNODE
using IncompressibleNavierStokes
using NeuralClosure
using OrdinaryDiffEqTsit5

@testset "CoupledNode integration" begin

    # Define parameters for the model
    nles = 16
    T = Float32
    N = nles + 2 # Do NOT forget to consider that our data include the bc
    D = 2
    rng = Xoshiro(123)
    r = [2, 2]
    c = [2, 2]
    σ = [tanh, identity]
    b = [true, false]
    emb_sizes = [4, 4]
    patch_sizes = [2, 2]
    n_heads = [2, 2]
    use_attention = [true, true]
    sum_attention = [false, false]

    # Create the model
    closure, θ_start, st = attentioncnn(
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

    # Define input tensor and pass through model
    batch = 4
    input_tensor = rand(T, N, N, D, batch)
    output = Lux.apply(closure, input_tensor, θ_start, st)[1]
    @test size(output) == size(input_tensor)


    # Read conf
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    conf = NS.read_config("./config.yaml")
    conf["params"]["backend"] = CPU()

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
    function namedtupleload(file)
        dict = load(file)
        k, v = keys(dict), values(dict)
        pairs = @. Symbol(k) => v
        (; pairs...)
    end
    data_train = []
    data_i = namedtupleload("data_train.jld2")
    push!(data_train, hcat(data_i))


    # Create the io array
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    io_train = NS.create_io_arrays_posteriori(data_train, setup)

    # Create the dataloader
    θ = device(copy(θ_start))
    nunroll = 2
    nunroll_valid = 2
    dataloader_post = NS.create_dataloader_posteriori(
        io_train[1];
        nunroll = nunroll,
        rng = Random.Xoshiro(24),
        device = device,
    )

    # Create the right hand side and the loss
    dudt_nn = NS.create_right_hand_side_with_closure(setup[1], psolver, closure, st)
    loss = CoupledNODE.create_loss_post_lux(
        dudt_nn;
        sciml_solver = Tsit5(),
        dt = T(conf["params"]["Δt"]),
        use_cuda = CUDA.functional(),
    )
    callbackstate = trainstate = nothing


    # For testing reason, explicitely set up the probelm
    # Notice that this is automatically done in CoupledNODE
    u, t = dataloader_post()
    griddims = Zygote.@ignore ((:) for _ = 1:(ndims(u)-2))
    x = u[griddims..., :, 1]
    y = u[griddims..., :, 2:end] # remember to discard sol at the initial time step
    tspan, dt, prob, pred = nothing, nothing, nothing, nothing # initialize variable outside allowscalar do.
    dt = @views t[2:2] .- t[1:1]
    dt = only(Array(dt))
    function get_tspan(t)
        return (Array(t)[1], Array(t)[end])
    end
    tspan = get_tspan(t)
    prob = ODEProblem(dudt_nn, x, tspan, θ)
    pred = Array(
        solve(prob, Tsit5(); u0 = x, p = θ, adaptive = false, saveat = Array(t), dt = dt),
    )

    # Test the forward pass
    @test size(pred[:, :, :, 2:end]) == size(y)


    # Test the backward pass
    p = prob.p
    y = prob.u0
    f = prob.f
    λ = zero(prob.u0)
    _dy, back = Zygote.pullback(y, p) do u, p
        vec(f(u, p, t))
    end
    tmp1, tmp2 = back(λ)
    @test size(tmp1) == (18, 18, 2)
    @test size(tmp2) == (1383262,)

    # Final integration test of the entire train interface
    l, trainstate = CoupledNODE.train(
        closure,
        θ,
        st,
        dataloader_post,
        loss;
        tstate = trainstate,
        nepochs = 2,
        #alg = OptimiserChain(Adam(T(1.0e-3)), ClipGrad(0.1)),
        alg = Adam(T(1.0e-3)),
        cpu = true,
    )
    @test isnan(l) == false
    @test trainstate.step == 2
    @test any(isnan, trainstate.parameters) == false

end
