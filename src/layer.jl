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
        #U = init_weight(rng, T, N * N * d, n_patches * n_heads * dh),
        U = init_weight(rng, T, emb_size, emb_size),  # i.e., 128 × 128
        # the positional embedding
        pos_emb = init_weight(rng, T, emb_size, div(N, patch_size), div(N, patch_size)),
        # and a final decoder
        dec = init_weight(rng, T, patch_size * patch_size * d, emb_size),  # (2738, 128)
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
    #size_U = N * N * d * n_patches * n_heads * dh
    size_U = emb_size * emb_size
    size_dec = patch_size * patch_size * d * emb_size
    size_pos_emb = emb_size * div(N, patch_size) * div(N, patch_size)
    total_size =
        size_wQ + size_wK + size_wV + size_Ew + size_Eb + size_U + size_dec + size_pos_emb
    return total_size
end
Lux.statelength(::attention) = 12

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
    emb_size = state.emb_size

    Ew = params.Ew
    Eb = params.Eb
    wQ = params.wQ
    wK = params.wK
    wV = params.wV
    U = params.U
    pos_emb = params.pos_emb
    dec = params.dec

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
    # Combine reshapes and matrix multiplications
    A_flat = reshape(A, n_heads * dh, :)  # (emb_size, np * batch)
    MSA = U * A_flat                     # Apply U (U ∈ (emb_size, emb_size)) -> (emb_size, np * batch)

    # (8) Decode each patch and reshape directly into the final image layout
    output = reshape(dec * MSA, ps, ps, d, num_patches_1d, num_patches_1d, batch)

    # (9) Reorder to reconstruct the full image
    output = permutedims(output, (1, 4, 2, 5, 3, 6))  # (ps, np1d, ps, np1d, d, batch)
    output = reshape(output, N, N, d, batch)           # (148, 148, 2, batch)

    # Attention layer does not modify state
    output, state

end
