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
    @info "types: x $(typeof(x)), W $(typeof(W)), y $(typeof(y))"

    # Define the kernel function
    @kernel inbounds=true function QKV_kernel!(y, x, W)
        h, p, d, b = @index(Global, NTuple)  # Get the global thread indices

        for i in 1:emb_size  # Contract over the embedding space
            @atomic y[h, p, d, b] += x[i, p, b] * W[h, d, i]
        end
    end

    QKV_kernel!(backend, workgroupsize)(y, x, W; ndrange = (n_heads, n_patches, dh, batch))
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
        @warn " I am in the qkv PB!"
        # Kernel to compute dX
        @kernel inbounds=true function grad_x_kernel!(dX, Δy, W)
            i, p, b = @index(Global, NTuple)
            for h in 1:n_heads, d in 1:dh
                @atomic dX[i, p, b] += Δy[h, p, d, b] * W[h, d, i]
            end
        end

        # Kernel to compute dW
        @kernel inbounds=true function grad_w_kernel!(dW, Δy, x)
            h, d, i = @index(Global, NTuple)
            for p in 1:n_patches, b in 1:batch
                @atomic dW[h, d, i] += Δy[h, p, d, b] * x[i, p, b]
            end
        end

        grad_x_kernel!(backend, workgroupsize)(dX, Δy, W; ndrange = size(dX))
        grad_w_kernel!(backend, workgroupsize)(dW, Δy, x; ndrange = size(dW))

        return NoTangent(), dX, dW
    end

    return y,QKV_pb 
end


function compute_attention(Q, K)
    n_heads, n_patches, dh, batch= size(Q)

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
    @kernel function attention_kernel!(A, Q, K)
        h, p, b = @index(Global, NTuple)  # Get the global thread indices

        # Perform the dot product of Q and K^T
        for i in 1:n_patches
            for j in 1:n_patches
                # Perform the dot product for the current head, patch, and batch
                A[h, i, j, b] = sum(Q[h, i, :, b] .* K[h, j, :, b])
            end
        end
    end

    # Run the kernel
    attention_kernel!(backend, workgroupsize)(A, Q, K; ndrange = (n_heads, n_patches, batch))

    return A
end

function ChainRulesCore.rrule(::typeof(compute_attention), Q, K)
    A = compute_attention(Q, K)  # Forward pass
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

    function compute_attention_pb(ΔA)
        @warn " I am in the attention PB!"

        # Define kernel for gradient wrt Q and K
        @kernel function attention_kernel_Qgrad!(dQ, K, ΔA)
            h, i, d, b = @index(Global, NTuple)  # Get the global thread indices
            
            for j in 1:n_patches
                # check indices if correct matematically
                @atomic dQ[h, i, d, b] += ΔA[h, i, j, b] * K[h, j, d, b]
            end
        end

        # Define kernel for gradient wrt Q and K
        @kernel function attention_kernel_Kgrad!(dK, Q, ΔA)
            h, j, d, b = @index(Global, NTuple)  # Get the global thread indices

            for i in 1:n_patches
                @atomic dK[h, j, d, b] += ΔA[h, i, j, b] * Q[h, i, d, b]
            end
        end
        # Run the kernel to compute gradients
        attention_kernel_Qgrad!(backend, workgroupsize)(dQ, K, ΔA; ndrange = (n_heads, n_patches, dh, batch))
        attention_kernel_Kgrad!(backend, workgroupsize)(dK, Q, ΔA; ndrange = (n_heads, n_patches, dh, batch))

        return NoTangent(), dQ, dK
    end
    return A, compute_attention_pb
end