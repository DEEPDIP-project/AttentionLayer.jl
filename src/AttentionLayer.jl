module AttentionLayer

using CUDA: CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

include("model.jl")

export create_CNO, create_CNOdownsampler, create_CNOupsampler, create_CNOactivation

end
