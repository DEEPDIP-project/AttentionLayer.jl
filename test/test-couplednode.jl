using Test
using Adapt
using Lux
using JLD2
using AttentionLayer: attention, attentioncnn
using ComponentArrays: ComponentArray
using Random
using Zygote: Zygote
using CoupledNODE
using IncompressibleNavierStokes
using NeuralClosure

@testset "CoupledNode integration" begin

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
    patch_sizes = [8, 5]
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
    batch = 16
    input_tensor = rand(T, N, N, D, batch)
    output = closure(input_tensor, θ_start, st)


    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    conf = NS.read_config("./config.yaml")
    conf["params"]["backend"] = CPU()

    params = NS.load_params(conf)
    device(x) = adapt(params.backend, x)
    nles = conf["params"]["nles"][1]

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

    # Read the data in the format expected by the CoupledNODE
    x = ntuple(α -> LinRange(T(0.0), T(1.0), N + 1), params.D)
    setup = [Setup(; x = x, Re = params.Re, params.backend)]

    ## Read the data in the format expected by the CoupledNODE
    #_comptime = rand(Float64)
    #N = nles
    #_c = rand(Float32, N+2, N+2, 2, 201)
    #_t = rand(Float32, 201)
    #_u = rand(Float32, N+2, N+2, 2, 201)

    ## Create a NamedTuple with the random content
    #data_train = [(comptime = _comptime, c = _c, t = _t, u = _u)]

    function namedtupleload(file)
        dict = load(file)
        k, v = keys(dict), values(dict)
        pairs = @. Symbol(k) => v
        (; pairs...)
    end
    data_train = []
    data_i = namedtupleload("data_train.jld2")
    push!(data_train, hcat(data_i))


    nunroll = 2
    nunroll_valid = 2

    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    io_train = NS.create_io_arrays_posteriori(data_train, setup)
    θ = device(copy(θ_start[1]))
    dataloader_post = NS.create_dataloader_posteriori(
        io_train[1];
        nunroll = nunroll,
        rng = Random.Xoshiro(24),
        device = device,
    )

    dudt_nn = NS.create_right_hand_side_with_closure(setup[1], psolver, closure, st)
    loss = create_loss_post_lux(
        dudt_nn;
        sciml_solver = Tsit5(),
        dt = dt,
        use_cuda = CUDA.functional(),
    )

    callbackstate = trainstate = nothing

    callbackstate, callback = NS.create_callback(
        closure,
        θ,
        io_train[1],
        loss,
        st;
        callbackstate = callbackstate,
        nunroll = nunroll_valid,
        rng = Xoshiro(24),
        device = device,
    )
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
        callback = callback,
    )

end
