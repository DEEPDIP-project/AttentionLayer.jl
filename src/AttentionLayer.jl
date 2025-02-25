module AttentionLayer

using CUDA: CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

include("layer.jl")
include("attention_cnn.jl")

end
