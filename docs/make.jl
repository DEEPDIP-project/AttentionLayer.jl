using AttentionLayer
using Documenter

DocMeta.setdocmeta!(
    AttentionLayer,
    :DocTestSetup,
    :(using AttentionLayer);
    recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [AttentionLayer],
    authors = "SCiarella <simoneciarella@gmail.com>",
    repo = "https://github.com/DEEPDIP-project/AttentionLayer.jl/blob/{commit}{path}#{line}",
    sitename = "AttentionLayer.jl",
    format = Documenter.HTML(;
        canonical = "https://DEEPDIP-project.github.io/AttentionLayer.jl",
    ),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/DEEPDIP-project/AttentionLayer.jl")
