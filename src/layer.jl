using Lux: Lux
using LuxCore: AbstractLuxLayer
using NNlib: batched_mul

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
    dh = div(emb_size, n_heads) # dimension of each head (scale down the embedding size)
    attention(T, N, d, emb_size, patch_size, n_patches, n_heads, dh, init_weight)
end

# We also need to specify how to initialize the parameters and states.

function Lux.initialparameters(
    rng::AbstractRNG,
    (; T, N, d, emb_size, patch_size, n_patches, dh, n_heads, init_weight)::attention,
)
    (;
        # the attention weights have this size
        wQ = init_weight(rng, T, n_heads, dh, emb_size),
        wK = init_weight(rng, T, n_heads, dh, emb_size),
        wV = init_weight(rng, T, n_heads, dh, emb_size),
        # then the embedding operator
        Ew = init_weight(rng, T, emb_size, patch_size * patch_size * d),
        Eb = zeros(T, emb_size),
        # then the multihead attention output matrix
        U = init_weight(rng, T, N * N * d, n_patches * n_heads * dh),
        # and the positional embedding
        pos_emb = init_weight(rng, T, emb_size, div(N, patch_size), div(N, patch_size)),
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
        n_patches = n_patches, # total number of patches
        n_heads = n_heads,
        dh = dh,
        sqrtDh = T(sqrt(dh)),
        num_patches_1d = div(N, patch_size),
    )
end
function Lux.parameterlength(
    (; N, d, n_heads, dh, emb_size, patch_size, n_patches)::attention,
)
    size_wQ = n_heads * dh * emb_size
    size_wK = n_heads * dh * emb_size
    size_wV = n_heads * dh * emb_size
    size_Ew = emb_size * patch_size * patch_size * d
    size_Eb = emb_size
    size_U = N * N * d * n_patches * n_heads * dh

    total_size = size_wQ + size_wK + size_wV + size_Ew + size_Eb + size_U
    return total_size
end
Lux.statelength(::attention) = 11

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
    num_patches_1d = state.num_patches_1d

    Ew = params.Ew
    Eb = params.Eb
    wQ = params.wQ
    wK = params.wK
    wV = params.wV
    U = params.U
    pos_emb = params.pos_emb

    batch = size(x, ndims(x))

    # (1) Split the image into patches
    x_patches = reshape(x, ps, num_patches_1d, ps, num_patches_1d, d, batch)
    x_patches = permutedims(x_patches, (1, 3, 5, 2, 4, 6))
    # (2) flatten the patches
    x_patches = reshape(x_patches, ps * ps * d, :)
    # (3) project the patches onto the embedding space
    x_emb = Ew * x_patches .+ Eb
    x_emb = reshape(x_emb, size(x_emb, 1), num_patches_1d, num_patches_1d, batch)

    # (4) add the positional embedding
    # notice that we use 1D positional embedding, as suggested [here](https://arxiv.org/pdf/2010.11929)
    x_lemb = x_emb .+ pos_emb
    x_lemb = reshape(x_lemb, size(x_lemb, 1), num_patches_1d * num_patches_1d, batch)

    # (5) compute the attention scores
    Q = compute_QKV(x_lemb, wQ)
    K = compute_QKV(x_lemb, wK)
    V = compute_QKV(x_lemb, wV)

    # (6) Compute attention scores
    A = attention_weights(Q, K)
    A = Lux.softmax(A / sqrtDh, dims = 3)
    A = attention_scores(A, V)

    # (7) multihead attention
    MSA = reshape(A, n_heads * np * dh, size(x, ndims(x)))
    MSA = U * MSA
    MSA = reshape(MSA, size(x)...)

    # Attention layer does not modify state
    MSA, state
end
