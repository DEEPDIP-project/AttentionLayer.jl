# AttentionLayer

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://DEEPDIP-project.github.io/AttentionLayer.jl/stable)
[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://DEEPDIP-project.github.io/AttentionLayer.jl/dev)
[![Build Status](https://github.com/DEEPDIP-project/AttentionLayer.jl/workflows/Test/badge.svg)](https://github.com/DEEPDIP-project/AttentionLayer.jl/actions)
[![Test workflow status](https://github.com/DEEPDIP-project/AttentionLayer.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/DEEPDIP-project/AttentionLayer.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Lint workflow Status](https://github.com/DEEPDIP-project/AttentionLayer.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/DEEPDIP-project/AttentionLayer.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/DEEPDIP-project/AttentionLayer.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/DEEPDIP-project/AttentionLayer.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/DEEPDIP-project/AttentionLayer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DEEPDIP-project/AttentionLayer.jl)
[![DOI](https://zenodo.org/badge/887387729.svg)](https://doi.org/10.5281/zenodo.14191587)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/DEEPDIP-project/AttentionLayer.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

This package implements the [attention mechanism](https://arxiv.org/abs/1706.03762) as a Lux layer.
It can then be used for [closure modeling](https://github.com/DEEPDIP-project/CoupledNODE.jl).

## Install

```julia
using Pkg
Pkg.add(url="git@github.com:DEEPDIP-project/AttentionLayer.jl.git")
```

## Usage

You are probably interested in using the `attentioncnn` model, which is a built-in cnn that uses the attention mechanism.
Here is an example of how to use it:

* first you define the parameters of the model

```julia
    T = Float32 # the type of the data
    N = 16 # size of the input
    D = 2 # number of channels
    rng = Xoshiro(123) # random number generator
    r = [2, 2] # radii of the attention mechanism
    c = [4, 2] # number of features of the intermediate layers
    σ = [tanh, identity] # activation functions
    b = [true, false] # use bias
    emb_sizes = [8, 8] # size of the embeddings
    patch_sizes = [8, 5] # size of the patches in which the attention mechanism is applied
    n_heads = [2, 2] # number of heads of the attention mechanism
    use_attention = [true, true] # use the attention at this layer
    sum_attention = [false, false] # use attention in sum mode instead of concat mode (BUG)
```

* then you can call the model

```julia
    closure, θ, st = attentioncnn(
        T = T,
        N = N,
        D = D,
        data_ch = D,
        radii = r,
        channels = c,
        activations = σ,
        use_bias = b,
        use_attention = use_attention,
        emb_sizes = emb_sizes,
        patch_sizes = patch_sizes,
        n_heads = n_heads,
        sum_attention = sum_attention,
        rng = rng,
        use_cuda = false,
    )
```

Look in `test/` for more examples about how to use the package.

## How to Cite

If you use AttentionLayer.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/DEEPDIP-project/AttentionLayer.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://DEEPDIP-project.github.io/AttentionLayer.jl/dev/90-contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SCiarella"><img src="https://avatars.githubusercontent.com/u/58949181?v=4?s=100" width="100px;" alt="SCiarella"/><br /><sub><b>SCiarella</b></sub></a><br /><a href="#code-SCiarella" title="Code">💻</a> <a href="#test-SCiarella" title="Tests">⚠️</a> <a href="#maintenance-SCiarella" title="Maintenance">🚧</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
