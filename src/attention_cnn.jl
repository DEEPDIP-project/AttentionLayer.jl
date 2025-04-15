using Lux
using NNlib: pad_circular
using Random: Random
using ComponentArrays: ComponentArray

"""
    attentioncnn(; T, D, data_ch, radii, channels, activations, use_bias, use_attention, emb_sizes, Ns, patch_sizes, n_heads, sum_attention, rng = Random.default_rng(), use_cuda)

Constructs a convolutional neural network model `closure(u, θ)` that predicts the commutator error (i.e. closure).
Before every convolutional layer, the input is augmented with the attention mechanism.

# Arguments
- `T`: The data type of the model (default: `Float32`).
- `D`: The data dimension.
- `data_ch`: The number of data channels (usually should be equal to `D`).
- `radii`: An array (size n_layers) with the radii of the kernels for the convolutional layers. Kernels will be symmetrical of size `2r+1`.
- `channels`: An array (size n_layers) with channel sizes for the convolutional layers.
- `activations`: An array (size n_layers) with activation functions for the convolutional layers.
- `use_bias`: An array (size n_layers) with booleans indicating whether to use bias in each convolutional layer.
- `use_attention`: A boolean indicating whether to use the attention mechanism.
- `emb_sizes`: An array (size n_layers) with the embedding sizes for the attention mechanism.
- `Ns`: The spatial dimension for all the attention layers.
- `patch_sizes`: An array (size n_layers) with the patch sizes for the attention mechanism.
- `n_heads`: An array (size n_layers) with the number of heads for the attention mechanism.
- `sum_attention`: An array (size n_layers) with booleans indicating whether to sum the attention output with the input.
- `rng`: A random number generator (default: `Random.default_rng()`).
- `use_cuda`: A boolean indicating whether to use CUDA (default: `false`).

# Returns
A tuple `(chain, params, state)` where
- `chain`: The constructed Lux.Chain model.
- `params`: The parameters of the model.
- `state`: The state of the model.
"""
function attentioncnn(;
    T = Float32,
    D,
    data_ch,
    radii,
    channels,
    activations,
    use_bias,
    use_attention,
    emb_sizes,
    Ns,
    patch_sizes,
    n_heads,
    sum_attention,
    rng = Random.default_rng(),
    use_cuda = false,
)
    r, c, σ, b = radii, channels, activations, use_bias

    if use_cuda
        dev = Lux.gpu_device()
    else
        dev = Lux.cpu_device()
    end

    @warn "*** AttentionCNN is using the following device: $(dev) "

    # Weight initializer
    glorot_uniform_T(rng::Random.AbstractRNG, dims...) = glorot_uniform(rng, T, dims...)

    @assert length(c) == length(r) == length(σ) == length(b) "The number of channels, radii, activations, and use_bias must match"
    @assert c[end] == D "The number of output channels must match the data dimension"

    # Put the data channels at the beginning
    c = [data_ch; c]

    # If we use attention, we need to account for an extra attention channel at every layer (except the last that is only the output value)
    @assert length(use_attention) == length(c) - 1 "The number of attention flags must match the number of layers -1"

    # Syver uses a (single!) padder layer instead of adding padding to the convolutional layers
    # This padding will circulary add dimensions equal to the sum of the radii so they can be shaved layer by layer
    padder = ntuple(α -> (u -> pad_circular(u, sum(r); dims = α)), D)

    for i = 1:length(Ns)
        #assert that Ns[i] is divisible by patch_sizes[i]
        @assert Ns[i] % patch_sizes[i] == 0 "Patch size must be a divisor of the spatial dimension"
    end

    # Create the convolutional block
    conv_block = ()
    for i in eachindex(r)
        layers = ()

        if use_attention[i]
            attention_layer = Lux.Chain(
                # Use a convolution to get data on 2 channels only (https://github.com/DEEPDIP-project/AttentionLayer.jl/issues/14)
                Conv(
                    ntuple(α -> 2r[i] + 1, D),
                    c[i] => 2,
                    σ[i];
                    use_bias = true,
                    init_weight = glorot_uniform_T,
                    pad = (ntuple(α -> 2r[i] + 1, D) .- 1) .÷ 2,
                ),
                u -> crop_center(u, Ns[i]),
                attention(Ns[i], 2, emb_sizes[i], patch_sizes[i], n_heads[i]; T = T),
            )
            skip_connection = Lux.SkipConnection(
                attention_layer,
                if sum_attention[i]
                    @error "Sum attention not implemented yet"
                    (x, y) -> uncrop_center_sum(x, y)
                else
                    (att, conv) -> begin
                        uncrop_center_concat(conv, att)
                    end
                end,
                name = "Attention $i",
            )
            layers = (layers..., skip_connection)
        end

        if use_attention[i] && !sum_attention[i]
            c_att = c[i] + 2
        else
            c_att = c[i]
        end

        conv_layer = Lux.Chain(
            Conv(
                ntuple(α -> 2r[i] + 1, D),
                c_att => c[i+1],
                σ[i];
                use_bias = b[i],
                init_weight = glorot_uniform_T,
            ),
        )
        layers = (layers..., conv_layer)

        conv_block = (conv_block..., layers)
    end


    # Create closure model
    layers = (collocate, padder, conv_block..., decollocate)
    chain = Chain(layers...)
    params, state = Lux.setup(rng, chain)
    state = state |> dev
    params = ComponentArray(params) |> dev
    (chain, params, state)
end

"""
Interpolate velocity components to volume centers.

TODO, D and dir can be parameters istead of arguments I think
"""
function interpolate(A, D, dir)
    (i, a) = A
    if i > D
        return a  # Nothing to interpolate for extra layers
    end
    staggered = a .+ circshift(a, ntuple(x -> x == i ? dir : 0, D))
    staggered ./ 2
end

function collocate(u)
    D = ndims(u) - 2
    slices = eachslice(u; dims = D + 1)
    staggered_slices = map(x -> interpolate(x, D, 1), enumerate(slices))
    stack(staggered_slices; dims = D + 1)
end

"""
Interpolate closure force from volume centers to volume faces.
"""
function decollocate(u)
    D = ndims(u) - 2
    slices = eachslice(u; dims = D + 1)
    staggered_slices = map(x -> interpolate(x, D, -1), enumerate(slices))
    stack(staggered_slices; dims = D + 1)
end

"""
Crop the center of the input array to the desired size.

# Arguments
- `x`: The input array of size `[N, N, d, b]`.
- `M`: The desired size of the output array.

# Returns
An array `y` of size `[M, M, d, b]` cropped from the center of `x`.
"""
function crop_center(x, M)
    s0 = size(x)
    D = ndims(x) - 2
    N = s0[1]

    start_idx = div(N - M, 2) + 1
    end_idx = start_idx + M - 1

    slices = eachslice(x; dims = D + 1)
    cropped_slices = map(x -> x[start_idx:end_idx, start_idx:end_idx, :], slices)
    stack(cropped_slices; dims = D + 1)
end

"""
Add the input array `y` to the center of the array `x`.

# Arguments
- `x`: The original array of size `[N, N, d, b]`.
- `y`: The array to be added to the center of `x`, of size `[M, M, d, b]`.

# Returns
An array `z` of size `[N, N, d, b]` with `y` added to the center of `x`.
"""
function uncrop_center(x, y)
    s0 = size(x)
    D = ndims(x) - 2
    N = s0[1]
    M = size(y, 1)

    start_idx = div(N - M, 2) + 1
    end_idx = start_idx + M - 1

    slices_x = eachslice(x; dims = D + 1)
    slices_y = eachslice(y; dims = D + 1)

    updated_slices = map(
        (sx, sy) -> begin
            sx[start_idx:end_idx, start_idx:end_idx, :] += sy
            sx
        end,
        slices_x,
        slices_y,
    )

    stack(updated_slices; dims = D + 1)
end


"""
Concatenate the input array `y` to the center of the array `x` along the third dimension.

# Arguments
- `x`: The original array of size `[N, N, d, b]`.
- `y`: The array to be concatenated to the center of `x`, of size `[M, M, d, b]`.

# Returns
An array `z` of size `[N, N, d + d', b]` with `y` concatenated to the center of `x` along the third dimension.
"""
function uncrop_center_concat(x, y)
    s0 = size(x)
    D = ndims(x) - 2
    N = s0[1]
    M = size(y, 1)

    if N == M
        return cat(x, y; dims = D + 1)
    end

    pad_size = div(N - M, 2)
    padded_y = pad_constant(y, (pad_size, pad_size, 0, 0), 0)

    cat(x, padded_y; dims = D + 1)
end
