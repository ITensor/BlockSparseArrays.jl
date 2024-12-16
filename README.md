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
using BlockArrays: BlockArrays, BlockedVector, Block, blockedrange
using BlockSparseArrays: BlockSparseArray, blockstoredlength
using Test: @test, @test_broken

function main()
  # Block dimensions
  i1 = [2, 3]
  i2 = [2, 3]

  i_axes = (blockedrange(i1), blockedrange(i2))

  function block_size(axes, block)
    return length.(getindex.(axes, Block.(block.n)))
  end

  # Data
  nz_blocks = Block.([(1, 1), (2, 2)])
  nz_block_sizes = [block_size(i_axes, nz_block) for nz_block in nz_blocks]
  nz_block_lengths = prod.(nz_block_sizes)

  # Blocks with contiguous underlying data
  d_data = BlockedVector(randn(sum(nz_block_lengths)), nz_block_lengths)
  d_blocks = [
    reshape(@view(d_data[Block(i)]), block_size(i_axes, nz_blocks[i])) for
    i in 1:length(nz_blocks)
  ]
  b = BlockSparseArray(nz_blocks, d_blocks, i_axes)

  @test blockstoredlength(b) == 2

  # Blocks with discontiguous underlying data
  d_blocks = randn.(nz_block_sizes)
  b = BlockSparseArray(nz_blocks, d_blocks, i_axes)

  @test blockstoredlength(b) == 2

  # Access a block
  @test b[Block(1, 1)] == d_blocks[1]

  # Access a zero block, returns a zero matrix
  @test b[Block(1, 2)] == zeros(2, 3)

  # Set a zero block
  a₁₂ = randn(2, 3)
  b[Block(1, 2)] = a₁₂
  @test b[Block(1, 2)] == a₁₂

  # Matrix multiplication
  # TODO: Fix this, broken.
  @test_broken b * b ≈ Array(b) * Array(b)

  permuted_b = permutedims(b, (2, 1))
  @test permuted_b isa BlockSparseArray
  @test permuted_b == permutedims(Array(b), (2, 1))

  @test b + b ≈ Array(b) + Array(b)
  @test b + b isa BlockSparseArray
  # TODO: Fix this, broken.
  @test_broken blockstoredlength(b + b) == 2

  scaled_b = 2b
  @test scaled_b ≈ 2Array(b)
  @test scaled_b isa BlockSparseArray

  # TODO: Fix this, broken.
  @test_broken reshape(b, ([4, 6, 6, 9],)) isa BlockSparseArray{<:Any,1}

  return nothing
end

main()
````

# BlockSparseArrays.jl and BlockArrays.jl interface

````julia
using BlockArrays: BlockArrays, Block
using BlockSparseArrays: BlockSparseArray

i1 = [2, 3]
i2 = [2, 3]
B = BlockSparseArray{Float64}(i1, i2)
B[Block(1, 1)] = randn(2, 2)
B[Block(2, 2)] = randn(3, 3)

# Minimal interface

# Specifies the block structure
@show collect.(BlockArrays.blockaxes(axes(B, 1)))

# Index range of a block
@show axes(B, 1)[Block(1)]

# Last index of each block
@show BlockArrays.blocklasts(axes(B, 1))

# Find the block containing the index
@show BlockArrays.findblock(axes(B, 1), 3)

# Retrieve a block
@show B[Block(1, 1)]
@show BlockArrays.viewblock(B, Block(1, 1))

# Check block bounds
@show BlockArrays.blockcheckbounds(B, 2, 2)
@show BlockArrays.blockcheckbounds(B, Block(2, 2))

# Derived interface

# Specifies the block structure
@show collect(Iterators.product(BlockArrays.blockaxes(B)...))

# Iterate over block views
@show sum.(BlockArrays.eachblock(B))

# Reshape into 1-d
# TODO: Fix this, broken.
# @show BlockArrays.blockvec(B)[Block(1)]

# Array-of-array view
@show BlockArrays.blocks(B)[1, 1] == B[Block(1, 1)]

# Access an index within a block
@show B[Block(1, 1)[1, 1]] == B[1, 1]
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

