using CUDA
using Random: AbstractRNG
using KernelAbstractions
using Atomix: @atomic
using ChainRulesCore

function compute_QKV(x, W)
    n_heads, dh, emb_size = size(W)
    emb_size, n_patches, batch = size(x)
    if x isa CuArray || (x isa SubArray && parent(x) isa CuArray)
        y = CUDA.zeros(eltype(x), n_heads, n_patches, dh, batch)
        backend = CUDABackend()
        workgroupsize = 256
    else
        y = zeros(eltype(x), n_heads, n_patches, dh, batch)
        backend = CPU()
        workgroupsize = 64
    end

    # Define the kernel function
    @kernel inbounds = true function QKV_kernel!(y, x, W, emb_size)
        h, p, d, b = @index(Global, NTuple)  # Get the global thread indices

        for i = 1:emb_size # Contract over the embedding space
            @atomic y[h, p, d, b] += x[i, p, b] * W[h, d, i]
        end
    end

    QKV_kernel!(backend, workgroupsize)(
        y,
        x,
        W,
        emb_size;
        ndrange = (n_heads, n_patches, dh, batch),
    )
    return y
end

function ChainRulesCore.rrule(::typeof(compute_QKV), x, W)
    n_heads, dh, emb_size = size(W)
    emb_size, n_patches, batch = size(x)
    y = compute_QKV(x, W)  # Forward pass
    if x isa CuArray || (x isa SubArray && parent(x) isa CuArray)
        dX = CUDA.zeros(eltype(x), size(x))
        dW = CUDA.zeros(eltype(W), size(W))
        backend = CUDABackend()
        workgroupsize = 256
    else
        dX = zeros(eltype(x), size(x))
        dW = zeros(eltype(W), size(W))
        backend = CPU()
        workgroupsize = 64
    end

    function QKV_pb(Δy)  # Backward pass
        # Kernel to compute dX
        @kernel inbounds = true function grad_x_kernel!(dX, Δy, W, n_heads, dh)
            i, p, b = @index(Global, NTuple)
            for h = 1:n_heads, d = 1:dh
                @atomic dX[i, p, b] += Δy[h, p, d, b] * W[h, d, i]
            end
        end

        # Kernel to compute dW
        @kernel inbounds = true function grad_w_kernel!(dW, Δy, x, n_patches, batch)
            h, d, i = @index(Global, NTuple)
            for p = 1:n_patches, b = 1:batch
                @atomic dW[h, d, i] += Δy[h, p, d, b] * x[i, p, b]
            end
        end

        grad_x_kernel!(backend, workgroupsize)(dX, Δy, W, n_heads, dh; ndrange = size(dX))
        grad_w_kernel!(backend, workgroupsize)(
            dW,
            Δy,
            x,
            n_patches,
            batch;
            ndrange = size(dW),
        )

        return NoTangent(), dX, dW
    end

    return y, QKV_pb
end


function attention_weights(Q, K)
    n_heads, n_patches, dh, batch = size(Q)

    # Initialize output tensor
    if Q isa CuArray || (Q isa SubArray && parent(Q) isa CuArray)
        A = CUDA.zeros(eltype(Q), n_heads, n_patches, n_patches, batch)
        backend = CUDABackend()
        workgroupsize = 256
    else
        A = zeros(eltype(Q), n_heads, n_patches, n_patches, batch)
        backend = CPU()
        workgroupsize = 64
    end

    # Define kernel to perform matrix multiplication (Q * K^T)
    @kernel function attention_kernel!(A, Q, K, n_patches)
        h, d, b = @index(Global, NTuple)  # Get the global thread indices

        # Perform the dot product of Q and K^T
        for i = 1:n_patches
            for j = 1:n_patches
                # Perform the dot product for the current head, patch, embedded coord, and batch
                @atomic A[h, i, j, b] += Q[h, i, d, b] * K[h, j, d, b]
            end
        end
    end

    # Run the kernel
    attention_kernel!(backend, workgroupsize)(
        A,
        Q,
        K,
        n_patches;
        ndrange = (n_heads, dh, batch),
    )

    return A
end

function ChainRulesCore.rrule(::typeof(attention_weights), Q, K)
    A = attention_weights(Q, K)  # Forward pass
    n_heads, n_patches, dh, batch = size(Q)

    # Initialize gradients
    if Q isa CuArray || (Q isa SubArray && parent(Q) isa CuArray)
        dQ = CUDA.zeros(eltype(Q), size(Q))
        dK = CUDA.zeros(eltype(K), size(K))
        backend = CUDABackend()
        workgroupsize = 256
    else
        dQ = zeros(eltype(Q), size(Q))
        dK = zeros(eltype(K), size(K))
        backend = CPU()
        workgroupsize = 64
    end

    function attention_weights_pb(ΔA)

        # Define kernel for gradient wrt Q and K
        @kernel function attention_kernel_Qgrad!(dQ, K, ΔA, n_patches)
            h, i, d, b = @index(Global, NTuple)  # Get the global thread indices

            for j = 1:n_patches
                @atomic dQ[h, i, d, b] += ΔA[h, i, j, b] * K[h, j, d, b]
            end
        end

        # Define kernel for gradient wrt Q and K
        @kernel function attention_kernel_Kgrad!(dK, Q, ΔA, n_patches)
            h, j, d, b = @index(Global, NTuple)  # Get the global thread indices

            for i = 1:n_patches
                @atomic dK[h, j, d, b] += ΔA[h, i, j, b] * Q[h, i, d, b]
            end
        end
        # Run the kernel to compute gradients
        attention_kernel_Qgrad!(backend, workgroupsize)(
            dQ,
            K,
            ΔA,
            n_patches;
            ndrange = (n_heads, n_patches, dh, batch),
        )
        attention_kernel_Kgrad!(backend, workgroupsize)(
            dK,
            Q,
            ΔA,
            n_patches;
            ndrange = (n_heads, n_patches, dh, batch),
        )

        return NoTangent(), dQ, dK
    end
    return A, attention_weights_pb
end


function attention_scores(A, V)
    n_heads, n_patches, dh, batch = size(V)
    if V isa CuArray || (V isa SubArray && parent(V) isa CuArray)
        SA = CUDA.zeros(eltype(A), n_heads, n_patches, dh, batch)
        backend = CUDABackend()
        workgroupsize = 256
    else
        SA = zeros(eltype(A), n_heads, n_patches, dh, batch)
        backend = CPU()
        workgroupsize = 64
    end

    #Define the kernel to compute the attention scores
    @kernel inbounds = true function attention_kernel!(SA, A, V, dh)
        h, p, d, b = @index(Global, NTuple)
        for i = 1:dh
            SA[h, p, d, b] = A[h, d, i, b] * V[h, p, i, b]
        end
    end

    attention_kernel!(backend, workgroupsize)(SA, A, V, dh; ndrange = size(SA))
    return SA
end

function ChainRulesCore.rrule(::typeof(attention_scores), A, V)
    n_heads, n_patches, dh, batch = size(V)
    SA = attention_scores(A, V)  # Forward pass
    if A isa CuArray || (A isa SubArray && parent(A) isa CuArray)
        dA = CUDA.zeros(eltype(A), size(A))
        dV = CUDA.zeros(eltype(V), size(V))
        backend = CUDABackend()
        workgroupsize = 256
    else
        dA = zeros(eltype(A), size(A))
        dV = zeros(eltype(V), size(V))
        backend = CPU()
        workgroupsize = 64
    end

    function attention_score_pb(ΔSA)  # Backward pass

        # Define the kernel to compute the gradients
        @kernel inbounds = true function attention_grad_kernel!(dA, dV, ΔSA, A, V, dh)
            h, p, d, b = @index(Global, NTuple)
            for i = 1:dh
                @atomic dA[h, d, i, b] += ΔSA[h, p, d, b] * V[h, p, i, b]
                @atomic dV[h, p, i, b] += ΔSA[h, p, d, b] * A[h, d, i, b]
            end
        end

        attention_grad_kernel!(backend, workgroupsize)(
            dA,
            dV,
            ΔSA,
            A,
            V,
            dh;
            ndrange = size(ΔSA),
        )

        return NoTangent(), dA, dV
    end

    return SA, attention_score_pb
end
