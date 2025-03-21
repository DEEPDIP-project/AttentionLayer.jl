using Lux: Lux
using LuxCore: AbstractLuxLayer
using Random: AbstractRNG

struct attention{F} <: AbstractLuxLayer
    T::Type
    N::Int
    d::Int
    emb_size::Int
    patch_size::Int
    n_patches::Int
    n_heads::Int
    dh::Int
    init_weight::F
end

function attention(
    N::Int,
    d::Int,
    emb_size::Int,
    patch_size::Int,
    n_heads::Int;
    T = Float32,
    init_weight = Lux.glorot_uniform,
)
    if d != 2
        @error "This implementation only supports 2D data on 2 channels (https://github.com/DEEPDIP-project/AttentionLayer.jl/issues/14)"
    end
    @assert N % patch_size == 0 "N must be divisible by patch_size"
    n_patches = (div(N, patch_size))^d
    dh = div(emb_size, n_heads)
    attention(T, N, d, emb_size, patch_size, n_patches, n_heads, dh, init_weight)
end

# We also need to specify how to initialize the parameters and states.

function Lux.initialparameters(
    rng::AbstractRNG,
    (; T, N, d, emb_size, patch_size, n_patches, dh, n_heads, init_weight)::attention,
)
    (;
        # the attention weights have this size
        wQ = init_weight(rng, T, n_heads, dh, emb_size + 1),
        wK = init_weight(rng, T, n_heads, dh, emb_size + 1),
        wV = init_weight(rng, T, n_heads, dh, emb_size + 1),
        # then the embedding operator
        Ew = init_weight(rng, T, emb_size, patch_size * patch_size * d),
        Eb = zeros(T, emb_size),
        # then the multihead attention
        U = init_weight(rng, T, N * N * d, n_patches * n_heads * dh),
    )
end

function Lux.initialstates(
    rng::AbstractRNG,
    (; T, N, d, emb_size, patch_size, n_patches, dh, n_heads)::attention,
)
    (;
        T = T,
        N = N,
        d = d,
        emb_size = emb_size,
        patch_size = patch_size,
        n_patches = n_patches,
        n_heads = n_heads,
        dh = dh,
        sqrtDh = T(sqrt(dh)),
    )
end
function Lux.parameterlength(
    (; N, d, n_heads, dh, emb_size, patch_size, n_patches)::attention,
)
    size_wQ = n_heads * dh * (emb_size + 1)
    size_wK = n_heads * dh * (emb_size + 1)
    size_wV = n_heads * dh * (emb_size + 1)
    size_Ew = emb_size * patch_size * patch_size * d
    size_Eb = emb_size
    size_U = N * N * d * n_patches * n_heads * dh

    total_size = size_wQ + size_wK + size_wV + size_Ew + size_Eb + size_U
    return total_size
end
Lux.statelength(::attention) = 9

# This is what each layer does:
# expected input shape: [N, N, d, batch]
# expected output shape: [N, N, d, batch]
function ((;)::attention)(x, params, state)
    N = state.N
    d = state.d
    np = state.n_patches
    ps = state.patch_size
    dh = state.dh
    sqrtDh = state.sqrtDh
    n_heads = state.n_heads

    Ew = params.Ew
    Eb = params.Eb
    wQ = params.wQ
    wK = params.wK
    wV = params.wV
    U = params.U

    # (1) Split the image into patches
    num_patches = div(N, ps)
    #The subarray of x here is by default a copy, but it can be a view (its not edited)
    x_patches = [
        @view(x[(i*ps+1):(i*ps+ps), (j*ps+1):(j*ps+ps), :, :]) for
        i = 0:(num_patches-1), j = 0:(num_patches-1)
    ]
    # (2) flatten the patches
    # reshape is fine and will not create a copy here, as only the first dims are merged, and because julia
    # is column order, this does not change the shape of the underlying data, this is true for all following reshapes
    x_pflat = [reshape(p, ps * ps * d, size(p, ndims(p))) for p in x_patches]

    # (3) project the patches onto the embedding space
    x_emb = [Ew * p .+ Eb for p in x_pflat]

    # (4) positional embedding
    # notice that we use 1D positional embedding, as suggested [here](https://arxiv.org/pdf/2010.11929)
    x_lemb = [
        cat(p, ones(state.T, 1, size(p)[2:end]...) * i; dims = 1) for
        (i, p) in enumerate(x_emb)
    ]

    # (5) compute the attention scores
    # [!] notice that you can not reuse some variable names otherwise Zygote gets confused
    Q0 = [wQ[i, :, :] * x_lemb[patchi] for i = 1:n_heads, patchi = 1:np]
    K0 = [wK[i, :, :] * x_lemb[patchi] for i = 1:n_heads, patchi = 1:np]
    V0 = [wV[i, :, :] * x_lemb[patchi] for i = 1:n_heads, patchi = 1:np]
    # Reshape Q, K, V to match desired output dimensions
    Q = reshape(vcat(Q0...), (n_heads, np, dh, size(x, ndims(x))))
    K = reshape(vcat(K0...), (n_heads, np, dh, size(x, ndims(x))))
    V = reshape(vcat(V0...), (n_heads, np, dh, size(x, ndims(x))))
    # (6) Compute attention scores without mutations
    A = [Lux.softmax(Q[i, p, :, :] .* K[i, p, :, :] / sqrtDh) for i = 1:n_heads, p = 1:np]
    A = reshape(vcat(A...), (n_heads, np, dh, size(x, ndims(x))))
    SA = A .* V

    # (7) multihead attention
    MSA = reshape(SA, n_heads * np * dh, size(x, ndims(x)))
    MSA = U * MSA
    MSA = reshape(MSA, size(x)...)

    # Attention layer does not modify state
    MSA, state
end
