# BlockSparseArrays.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/BlockSparseArrays.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/BlockSparseArrays.jl/dev/)
[![Build Status](https://github.com/ITensor/BlockSparseArrays.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/BlockSparseArrays.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/BlockSparseArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/BlockSparseArrays.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A block sparse array type in Julia based on the [`BlockArrays.jl`](https://github.com/JuliaArrays/BlockArrays.jl) interface.

## Installation instructions

This package resides in the `ITensor/ITensorRegistry` local registry.
In order to install, simply add that registry through your package manager.
This step is only required once.
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
or:
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

Then, the package can be added as usual through the package manager:

```julia
julia> Pkg.add("BlockSparseArrays")
```

## Examples

````julia
using BlockArrays: Block
using BlockSparseArrays: BlockSparseArray, blockstoredlength
using Test: @test

a = BlockSparseArray{Float64}(undef, [2, 3], [2, 3])
a[Block(1, 2)] = randn(2, 3)
a[Block(2, 1)] = randn(3, 2)
@test blockstoredlength(a) == 2
b = a .+ 2 .* a'
@test Array(b) ≈ Array(a) + 2 * Array(a')
@test blockstoredlength(b) == 2
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

