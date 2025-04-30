module AttentionLayer

using CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array
using Random: Random, AbstractRNG
using KernelAbstractions

include("utils.jl")
include("layer.jl")
include("attention_cnn.jl")

end
