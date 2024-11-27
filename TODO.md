- Add Aqua tests.
- Turn the package extensions into actual package extensions:
  - BlockSparseArraysAdaptExt
  - BlockSparseArraysGradedUnitRangesExt
  - BlockSparseArraysTensorAlgebraExt

# Proposals for interfaces based on `BlockArrays.jl`, `SparseArrays`, and `BlockSparseArrays.jl`

```julia
# BlockSparseArray interface

# Define `eachblockindex`
eachblockindex(B::BlockArrays.AbstractBlockArray) = Iterators.product(BlockArrays.blockaxes(B)...)

eachblockindex(B::BlockArrays.AbstractBlockArray, b::Block) # indices in a block

blocksize(B::BlockArrays.AbstractBlockArray, b::Block) # size of a block
blocksize(axes, b::Block) # size of a block

blocklength(B::BlockArrays.AbstractBlockArray, b::Block) # length of a block
blocklength(axes, b::Block) # length of a block

# Other functions
BlockArrays.blocksize(B) # number of blocks in each dimension
BlockArrays.blocksizes(B) # length of blocks in each dimension

tuple_block(Block(2, 2)) == (Block(2), Block(2)) # Block.(b.n)
blocksize(axes, b::Block) = map(axis -> length(axis[Block(b.n)]), axes)
blocksize(B, Block(2, 2)) = size(B[Block(2, 2)]) # size of a specified block

# SparseArrays interface

findnz(S) # outputs nonzero keys and values (SparseArrayKit.nonzero_pairs)
nonzeros(S) # vector of structural nonzeros (SparseArrayKit.nonzero_values)
nnz(S) # number of nonzero values (SparseArrayKit.nonzero_length)
rowvals(S) # row that each nonzero value in `nonzeros(S)` is in
nzrange(S, c) # range of linear indices into `nonzeros(S)` for values in column `c`
findall(!iszero, S) # CartesianIndices of numerical nonzeros
issparse(S)
sparse(A) # convert to sparse
dropzeros!(S)
droptol!(S, tol)

# BlockSparseArrays.jl + SparseArrays

blockfindnz(B) # outputs nonzero block indices/keys and block views
blocknonzeros(B)
blocknnz(S)
blockfindall(!iszero, B)
isblocksparse(B)
blocksparse(A)
blockdropzeros!(B)
blockdroptol!(B, tol)

# SparseArrayKit.jl interface

nonzero_pairs(a) # SparseArrays.findnz
nonzero_keys(a) # SparseArrays.?
nonzero_values(a) # SparseArrays.nonzeros
nonzero_length(a) # SparseArrays.nnz

# BlockSparseArrays.jl + SparseArrayKit.jl interface

block_nonzero_pairs
block_nonzero_keys
block_nonzero_values
block_nonzero_length
```
