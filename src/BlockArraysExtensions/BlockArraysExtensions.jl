using ArrayLayouts: ArrayLayouts, MemoryLayout, sub_materialize
using BlockArrays:
  BlockArrays,
  AbstractBlockArray,
  AbstractBlockVector,
  Block,
  BlockIndex,
  BlockIndexRange,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedOneTo,
  BlockedUnitRange,
  BlockedVector,
  block,
  blockaxes,
  blockedrange,
  blockindex,
  blocks,
  findblock,
  findblockindex
using Dictionaries: Dictionary, Indices
using SparseArraysBase:
  SparseArraysBase,
  eachstoredindex,
  getunstoredindex,
  isstored,
  setunstoredindex!,
  storedlength

# A return type for `blocks(array)` when `array` isn't blocked.
# Represents a vector with just that single block.
struct SingleBlockView{N,Array<:AbstractArray{<:Any,N}} <: AbstractArray{Array,N}
  array::Array
end
Base.parent(a::SingleBlockView) = a.array
Base.size(a::SingleBlockView) = ntuple(Returns(1), ndims(a))
blocks_maybe_single(a) = blocks(a)
blocks_maybe_single(a::Array) = SingleBlockView(a)
function Base.getindex(a::SingleBlockView{N}, index::Vararg{Int,N}) where {N}
  @assert all(isone, index)
  return parent(a)
end

# A wrapper around a potentially blocked array that is not blocked.
struct NonBlockedArray{T,N,Array<:AbstractArray{T,N}} <: AbstractArray{T,N}
  array::Array
end
Base.parent(a::NonBlockedArray) = a.array
Base.size(a::NonBlockedArray) = size(parent(a))
Base.getindex(a::NonBlockedArray{<:Any,N}, I::Vararg{Integer,N}) where {N} = parent(a)[I...]
# Views of `NonBlockedArray`/`NonBlockedVector` are eager.
# This fixes an issue in Julia 1.11 where reindexing defaults to using views.
# TODO: Maybe reconsider this design, and allows views to work in slicing.
Base.view(a::NonBlockedArray, I...) = a[I...]
BlockArrays.blocks(a::NonBlockedArray) = SingleBlockView(parent(a))
const NonBlockedVector{T,Array} = NonBlockedArray{T,1,Array}
NonBlockedVector(array::AbstractVector) = NonBlockedArray(array)

# BlockIndices works around an issue that the indices of BlockSlice
# are restricted to AbstractUnitRange{Int}.
struct BlockIndices{B,T<:Integer,I<:AbstractVector{T}} <: AbstractVector{T}
  blocks::B
  indices::I
end
for f in (:axes, :unsafe_indices, :axes1, :first, :last, :size, :length, :unsafe_length)
  @eval Base.$f(S::BlockIndices) = Base.$f(S.indices)
end
Base.getindex(S::BlockIndices, i::Integer) = getindex(S.indices, i)

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices::BlockIndices)
  # TODO: Is this a good definition? It ignores `indices.indices`.
  return a[indices.blocks]
end

# Generalization of to `BlockArrays._blockslice`:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/v1.6.3/src/views.jl#L13-L14
# Used by `BlockArrays.unblock`, which is used in `to_indices`
# to convert relative blockwise slices to absolute slices, but in a way
# that preserves the original relative blockwise slice information.
# TODO: Ideally this would be handled in BlockArrays.jl
# once slicing like `A[Block(1)[[1, 2]]]` is supported.
function _blockslice(x, y::AbstractUnitRange)
  return BlockSlice(x, y)
end
function _blockslice(x, y::AbstractVector)
  return BlockIndices(x, y)
end

# TODO: Constrain the type of `BlockIndices` more, this seems
# to assume that `S.blocks` is a list of blocks as opposed to
# a flat list of block indices like the definition below.
function Base.getindex(S::BlockIndices, i::BlockSlice{<:Block{1}})
  # TODO: Check that `i.indices` is consistent with `S.indices`.
  # It seems like this isn't handling the case where `i` is a
  # subslice of a block correctly (i.e. it ignores `i.indices`).
  @assert length(S.indices[Block(i)]) == length(i.indices)
  return _blockslice(S.blocks[Int(Block(i))], S.indices[Block(i)])
end

function Base.getindex(
  S::BlockIndices{<:AbstractBlockVector{<:BlockIndex{1}}}, i::BlockSlice{<:Block{1}}
)
  @assert length(S.indices[Block(i)]) == length(i.indices)
  return _blockslice(S.blocks[Block(i)], S.indices[Block(i)])
end

# This is used in slicing like:
# a = BlockSparseArray{Float64}([2, 2, 2, 2], [2, 2, 2, 2])
# I = BlockedVector([Block(4), Block(3), Block(2), Block(1)], [2, 2])
# a[I, I]
function Base.getindex(
  S::BlockIndices{<:AbstractBlockVector{<:Block{1}}}, i::BlockSlice{<:Block{1}}
)
  # TODO: Check for conistency of indices.
  # Wrapping the indices in `NonBlockedVector` reinterprets the blocked indices
  # as a single block, since the result shouldn't be blocked.
  return NonBlockedVector(BlockIndices(S.blocks[Block(i)], S.indices[Block(i)]))
end
function Base.getindex(
  S::BlockIndices{<:BlockedVector{<:Block{1},<:BlockRange{1}}}, i::BlockSlice{<:Block{1}}
)
  return i
end
# Views of `BlockIndices` are eager.
# This fixes an issue in Julia 1.11 where reindexing defaults to using views.
Base.view(S::BlockIndices, i) = S[i]

# Used in indexing such as:
# ```julia
# a = BlockSparseArray{Float64}([2, 2, 2, 2], [2, 2, 2, 2])
# I = BlockedVector([Block(4), Block(3), Block(2), Block(1)], [2, 2])
# b = @view a[I, I]
# @view b[Block(1, 1)[1:2, 2:2]]
# ```
# This is similar to the definition:
# @interface interface(a) to_indices(a, inds, I::Tuple{UnitRange{<:Integer},Vararg{Any}})
function Base.getindex(
  a::NonBlockedVector{<:Integer,<:BlockIndices}, I::UnitRange{<:Integer}
)
  ax = only(axes(parent(a).indices))
  brs = to_blockindices(ax, I)
  inds = blockedunitrange_getindices(ax, I)
  return NonBlockedVector(parent(a)[BlockSlice(brs, inds)])
end

function Base.getindex(S::BlockIndices, i::BlockSlice{<:BlockRange{1}})
  # TODO: Check that `i.indices` is consistent with `S.indices`.
  # TODO: Turn this into a `blockedunitrange_getindices` definition.
  subblocks = S.blocks[Int.(i.block)]
  subindices = mortar(
    map(1:length(i.block)) do I
      r = blocks(i.indices)[I]
      return S.indices[first(r)]:S.indices[last(r)]
    end,
  )
  return BlockIndices(subblocks, subindices)
end

# Used when performing slices like:
# @views a[[Block(2), Block(1)]][2:4, 2:4]
function Base.getindex(S::BlockIndices, i::BlockSlice{<:BlockVector{<:BlockIndex{1}}})
  subblocks = mortar(
    map(blocks(i.block)) do br
      return S.blocks[Int(Block(br))][only(br.indices)]
    end,
  )
  subindices = mortar(
    map(blocks(i.block)) do br
      S.indices[br]
    end,
  )
  return BlockIndices(subblocks, subindices)
end

# Similar to the definition of `BlockArrays.BlockSlices`:
# ```julia
# const BlockSlices = Union{Base.Slice,BlockSlice{<:BlockRange{1}}}
# ```
# but includes `BlockIndices`, where the blocks aren't contiguous.
const BlockSliceCollection = Union{
  Base.Slice,
  BlockSlice{<:Block{1}},
  BlockSlice{<:BlockRange{1}},
  BlockIndices{<:Vector{<:Block{1}}},
}
const BlockIndexRangeSlice = BlockSlice{
  <:BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}}
}
const BlockIndexRangeSlices = BlockIndices{
  <:BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}}
}
const BlockIndexVectorSlices = BlockIndices{
  <:BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexVector}}
}
const GenericBlockIndexVectorSlices = BlockIndices{
  <:BlockVector{<:GenericBlockIndex{1},<:Vector{<:BlockIndexVector}}
}
const SubBlockSliceCollection = Union{
  Base.Slice,
  BlockIndexRangeSlice,
  BlockIndexRangeSlices,
  BlockIndexVectorSlices,
  GenericBlockIndexVectorSlices,
}

# TODO: This is type piracy. This is used in `reindex` when making
# views of blocks of sliced block arrays, for example:
# ```julia
# a = BlockSparseArray{elt}(undef, ([2, 3], [2, 3]))
# b = @view a[[Block(1)[1:1], Block(2)[1:2]], [Block(1)[1:1], Block(2)[1:2]]]
# b[Block(1, 1)]
# ```
# Without this change, BlockArrays has the slicing behavior:
# ```julia
# julia> mortar([Block(1)[1:1], Block(2)[1:2]])[BlockSlice(Block(2), 2:3)]
# 2-element Vector{BlockIndex{1, Tuple{Int64}, Tuple{Int64}}}:
#  Block(2)[1]
#  Block(2)[2]
# ```
# while with this change it has the slicing behavior:
# ```julia
# julia> mortar([Block(1)[1:1], Block(2)[1:2]])[BlockSlice(Block(2), 2:3)]
# Block(2)[1:2]
# ```
# i.e. it preserves the types of the blocks better. Upstream this fix to
# BlockArrays.jl. Also consider overloading `reindex` so that it calls
# a custom `getindex` function to avoid type piracy in the meantime.
# Also fix this in BlockArrays:
# ```julia
# julia> mortar([Block(1)[1:1], Block(2)[1:2]])[Block(2)]
# 2-element Vector{BlockIndex{1, Tuple{Int64}, Tuple{Int64}}}:
#  Block(2)[1]
#  Block(2)[2]
# ```
function Base.getindex(
  a::BlockVector{<:BlockIndex{1},<:AbstractVector{<:BlockIndexRange{1}}},
  I::BlockSlice{<:Block{1}},
)
  # Check that the block slice corresponds to the correct block.
  @assert I.indices == only(axes(a))[Block(I)]
  return blocks(a)[Int(Block(I))]
end

# TODO: Use `Tuple` conversion once
# BlockArrays.jl PR is merged.
block_to_cartesianindex(b::Block) = CartesianIndex(b.n)

function blocks_to_cartesianindices(i::Indices{<:Block})
  return block_to_cartesianindex.(i)
end

function blocks_to_cartesianindices(d::Dictionary{<:Block})
  return Dictionary(blocks_to_cartesianindices(eachindex(d)), d)
end

function blockreshape(a::AbstractArray, dims::Tuple{Vector{Int},Vararg{Vector{Int}}})
  return blockreshape(a, blockedrange.(dims))
end
function blockreshape(a::AbstractArray, dim1::Vector{Int}, dim_rest::Vararg{Vector{Int}})
  return blockreshape(a, (dim1, dim_rest...))
end
# Fix ambiguity error.
function blockreshape(a::AbstractArray)
  return blockreshape(a, ())
end

tuple_oneto(n) = ntuple(identity, n)

function _blockreshape(a::AbstractArray, axes::Tuple{Vararg{AbstractUnitRange}})
  reshaped_blocks_a = reshape(blocks(a), blocklength.(axes))
  function f(I)
    block_axes_I = map(ntuple(identity, length(axes))) do i
      return Base.axes1(axes[i][Block(I[i])])
    end
    # TODO: Better converter here.
    return reshape(reshaped_blocks_a[I], block_axes_I)
  end
  bs = Dict(Block(Tuple(I)) => f(I) for I in eachstoredindex(reshaped_blocks_a))
  return blocksparse(bs, axes)
end

function blockreshape(
  a::AbstractArray, axes::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}}
)
  return _blockreshape(a, axes)
end
# Fix ambiguity error.
function blockreshape(a::AbstractArray, axes::Tuple{})
  return _blockreshape(a, axes)
end

function blockreshape(
  a::AbstractArray, axis1::AbstractUnitRange, axes_rest::Vararg{AbstractUnitRange}
)
  return blockreshape(a, (axis1, axes_rest...))
end

function cartesianindices(axes::Tuple, b::Block)
  return CartesianIndices(ntuple(dim -> axes[dim][Tuple(b)[dim]], length(axes)))
end

# Get the range within a block.
function blockindexrange(axis::AbstractUnitRange, r::AbstractUnitRange)
  bi1 = findblockindex(axis, first(r))
  bi2 = findblockindex(axis, last(r))
  b = block(bi1)
  # Range must fall within a single block.
  @assert b == block(bi2)
  i1 = blockindex(bi1)
  i2 = blockindex(bi2)
  return b[i1:i2]
end

function blockindexrange(
  axes::Tuple{Vararg{AbstractUnitRange,N}}, I::CartesianIndices{N}
) where {N}
  brs = blockindexrange.(axes, I.indices)
  b = Block(block.(brs))
  rs = map(br -> only(br.indices), brs)
  return b[rs...]
end

function blockindexrange(a::AbstractArray, I::CartesianIndices)
  return blockindexrange(axes(a), I)
end

# Get the blocks the range spans across.
function blockrange(axis::AbstractUnitRange, r::UnitRange)
  return findblock(axis, first(r)):findblock(axis, last(r))
end

# Occurs when slicing with `a[2:4, 2:4]`.
function blockrange(axis::BlockedOneTo{<:Integer}, r::BlockedUnitRange{<:Integer})
  # TODO: Check the blocks are commensurate.
  return findblock(axis, first(r)):findblock(axis, last(r))
end

function blockrange(axis::AbstractUnitRange, r::Int)
  ## return findblock(axis, r)
  return error("Slicing with integer values isn't supported.")
end

# This handles changing the blocking, for example:
# a = BlockSparseArray{Float64}([2, 2, 2, 2], [2, 2, 2, 2])
# I = blockedrange([4, 4])
# a[I, I]
# TODO: Generalize to `AbstractBlockedUnitRange`.
function blockrange(axis::BlockedOneTo{<:Integer}, r::BlockedOneTo{<:Integer})
  # TODO: Probably this is incorrect and should be something like:
  # return findblock(axis, first(r)):findblock(axis, last(r))
  return only(blockaxes(r))
end

# This handles block merging:
# a = BlockSparseArray{Float64}([2, 2, 2, 2], [2, 2, 2, 2])
# I = BlockedVector(Block.(1:4), [2, 2])
# I = BlockVector(Block.(1:4), [2, 2])
# I = BlockedVector([Block(4), Block(3), Block(2), Block(1)], [2, 2])
# I = BlockVector([Block(4), Block(3), Block(2), Block(1)], [2, 2])
# a[I, I]
function blockrange(axis::AbstractUnitRange, r::AbstractBlockVector{<:Block{1}})
  for b in r
    @assert b ∈ blockaxes(axis, 1)
  end
  return only(blockaxes(r))
end

function blockrange(axis::AbstractUnitRange, r::AbstractVector{<:Block{1}})
  for b in r
    @assert b ∈ blockaxes(axis, 1)
  end
  return r
end

using BlockArrays: BlockSlice
function blockrange(axis::AbstractUnitRange, r::BlockSlice)
  return blockrange(axis, r.block)
end

function blockrange(a::AbstractUnitRange, r::BlockIndices)
  return blockrange(a, r.blocks)
end

function blockrange(axis::AbstractUnitRange, r::Block{1})
  return r:r
end

function blockrange(axis::AbstractUnitRange, r::BlockIndexRange)
  return Block(r):Block(r)
end

function blockrange(axis::AbstractUnitRange, r::AbstractVector{<:BlockIndexRange{1}})
  return error("Slicing not implemented for range of type `$(typeof(r))`.")
end

function blockrange(
  axis::AbstractUnitRange,
  r::BlockVector{<:BlockIndex{1},<:AbstractVector{<:BlockIndexRange{1}}},
)
  return map(Block, blocks(r))
end

# This handles slicing with `:`/`Colon()`.
function blockrange(axis::AbstractUnitRange, r::Base.Slice)
  # TODO: Maybe use `BlockRange`, but that doesn't output
  # the same thing.
  return only(blockaxes(axis))
end

function blockrange(axis::AbstractUnitRange, r::NonBlockedVector)
  return Block.(Base.OneTo(1))
end

function blockrange(axis::AbstractUnitRange, r::AbstractVector{<:Integer})
  return Block.(Base.OneTo(1))
end

function blockrange(axis::AbstractUnitRange, r::BlockIndexVector)
  return Block(r):Block(r)
end

function blockrange(
  axis::AbstractUnitRange,
  r::BlockVector{<:BlockIndex{1},<:AbstractVector{<:BlockIndexVector}},
)
  return map(Block, blocks(r))
end

function blockrange(
  axis::AbstractUnitRange,
  r::BlockVector{<:GenericBlockIndex{1},<:AbstractVector{<:BlockIndexVector}},
)
  return map(Block, blocks(r))
end

function blockrange(axis::AbstractUnitRange, r)
  return error("Slicing not implemented for range of type `$(typeof(r))`.")
end

# This takes a range of indices `indices` of array `a`
# and maps it to the range of indices within block `block`.
function blockindices(a::AbstractArray, block::Block, indices::Tuple)
  return blockindices(axes(a), block, indices)
end

function blockindices(axes::Tuple, block::Block, indices::Tuple)
  return blockindices.(axes, Tuple(block), indices)
end

function blockindices(axis::AbstractUnitRange, block::Block, indices::AbstractUnitRange)
  indices_within_block = intersect(indices, axis[block])
  if iszero(length(indices_within_block))
    # Falls outside of block
    return 1:0
  end
  return only(blockindexrange(axis, indices_within_block).indices)
end

# This catches the case of `Vector{<:Block{1}}`.
# `BlockRange` gets wrapped in a `BlockSlice`, which is handled properly
#  by the version with `indices::AbstractUnitRange`.
#  TODO: This should get fixed in a better way inside of `BlockArrays`.
function blockindices(
  axis::AbstractUnitRange, block::Block, indices::AbstractVector{<:Block{1}}
)
  if block ∉ indices
    # Falls outside of block
    return 1:0
  end
  return Base.OneTo(length(axis[block]))
end

function blockindices(a::AbstractUnitRange, b::Block, r::BlockIndices)
  return blockindices(a, b, r.blocks)
end

function blockindices(
  a::AbstractUnitRange,
  b::Block,
  r::BlockVector{<:BlockIndex{1},<:AbstractVector{<:BlockIndexRange{1}}},
)
  # TODO: Change to iterate over `BlockRange(r)`
  # once https://github.com/JuliaArrays/BlockArrays.jl/issues/404
  # is fixed.
  for bl in blocks(r)
    if b == Block(bl)
      return only(bl.indices)
    end
  end
  return error("Block not found.")
end

function cartesianindices(a::AbstractArray, b::Block)
  return cartesianindices(axes(a), b)
end

# Output which blocks of `axis` are contained within the unit range `range`.
# The start and end points must match.
function findblocks(axis::AbstractUnitRange, range::AbstractUnitRange)
  # TODO: Add a test that the start and end points of the ranges match.
  return findblock(axis, first(range)):findblock(axis, last(range))
end

_block(indices) = block(indices)
_block(indices::CartesianIndices) = Block(ntuple(Returns(1), ndims(indices)))

function combine_axes(as::Vararg{Tuple})
  @assert allequal(length.(as))
  ndims = length(first(as))
  return ntuple(ndims) do dim
    dim_axes = map(a -> a[dim], as)
    return reduce(BlockArrays.combine_blockaxes, dim_axes)
  end
end

# Returns `BlockRange`
# Convert the block of the axes to blocks of the subaxes.
function subblocks(axes::Tuple, subaxes::Tuple, block::Block)
  @assert length(axes) == length(subaxes)
  return BlockRange(
    ntuple(length(axes)) do dim
      findblocks(subaxes[dim], axes[dim][Tuple(block)[dim]])
    end,
  )
end

# Returns `Vector{<:Block}`
function subblocks(axes::Tuple, subaxes::Tuple, blocks)
  return mapreduce(vcat, blocks; init=eltype(blocks)[]) do block
    return vec(subblocks(axes, subaxes, block))
  end
end

# Returns `Vector{<:CartesianIndices}`
function blocked_cartesianindices(axes::Tuple, subaxes::Tuple, blocks)
  return map(subblocks(axes, subaxes, blocks)) do block
    return cartesianindices(subaxes, block)
  end
end

# Represents a view of a block of a blocked array.
struct BlockView{T,N,Array<:AbstractArray{T,N}} <: AbstractArray{T,N}
  array::Array
  block::Tuple{Vararg{Block{1,Int},N}}
end
Base.parent(a::BlockView) = a.array
function Base.axes(a::BlockView)
  # TODO: Try to avoid conversion to `Base.OneTo{Int}`, or just convert
  # the element type to `Int` with `Int.(...)`.
  # When the axes of `parent(a)` are `GradedOneTo`, the block is `LabelledUnitRange`,
  # which has element type `LabelledInteger`. That causes conversion problems
  # in some generic Base Julia code, for example when printing `BlockView`.
  return ntuple(ndims(a)) do dim
    return Base.OneTo{Int}(only(axes(axes(parent(a), dim)[a.block[dim]])))
  end
end
function Base.size(a::BlockView)
  return length.(axes(a))
end
function Base.getindex(a::BlockView{<:Any,N}, index::Vararg{Int,N}) where {N}
  return blocks(parent(a))[Int.(a.block)...][index...]
end
function Base.setindex!(a::BlockView{<:Any,N}, value, index::Vararg{Int,N}) where {N}
  I = Int.(a.block)
  if !isstored(blocks(parent(a)), I...)
    unstored_value = getunstoredindex(blocks(parent(a)), I...)
    setunstoredindex!(blocks(parent(a)), unstored_value, I...)
  end
  blocks(parent(a))[I...][index...] = value
  return a
end

function SparseArraysBase.storedlength(a::BlockView)
  # TODO: Store whether or not the block is stored already as
  # a Bool in `BlockView`.
  I = CartesianIndex(Int.(a.block))
  # TODO: Use `eachblockstoredindex`.
  if I ∈ eachstoredindex(blocks(parent(a)))
    return storedlength(blocks(parent(a))[I])
  end
  return 0
end

## # Allow more fine-grained control:
## function ArrayLayouts.sub_materialize(layout, a::BlockView, ax)
##   return blocks(parent(a))[Int.(a.block)...]
## end
## function ArrayLayouts.sub_materialize(layout, a::BlockView)
##   return sub_materialize(layout, a, axes(a))
## end
## function ArrayLayouts.sub_materialize(a::BlockView)
##   return sub_materialize(MemoryLayout(a), a)
## end
function ArrayLayouts.sub_materialize(a::BlockView)
  return blocks(parent(a))[Int.(a.block)...]
end

function view!(a::AbstractArray{<:Any,N}, index::Block{N}) where {N}
  return view!(a, Tuple(index)...)
end
function view!(a::AbstractArray{<:Any,N}, index::Vararg{Block{1},N}) where {N}
  blocks(a)[Int.(index)...] = blocks(a)[Int.(index)...]
  return blocks(a)[Int.(index)...]
end
# Fix ambiguity error.
function view!(a::AbstractArray{<:Any,0})
  blocks(a)[] = blocks(a)[]
  return blocks(a)[]
end

function view!(a::AbstractArray{<:Any,N}, index::BlockIndexRange{N}) where {N}
  # TODO: Is there a better code pattern for this?
  indices = ntuple(N) do dim
    return Tuple(Block(index))[dim][index.indices[dim]]
  end
  return view!(a, indices...)
end
function view!(a::AbstractArray{<:Any,N}, index::Vararg{BlockIndexRange{1},N}) where {N}
  b = view!(a, Block.(index)...)
  r = map(index -> only(index.indices), index)
  return @view b[r...]
end

using MacroTools: @capture
is_getindex_expr(expr::Expr) = (expr.head === :ref)
is_getindex_expr(x) = false
macro view!(expr)
  if !is_getindex_expr(expr)
    error("@view must be used with getindex syntax (as `@view! a[i,j,...]`)")
  end
  @capture(expr, array_[indices__])
  return :(view!($(esc(array)), $(esc.(indices)...)))
end

# SVD additions
# -------------
using LinearAlgebra: Algorithm
using BlockArrays: BlockedMatrix

# svd first calls `eigencopy_oftype` to create something that can be in-place SVD'd
# Here, we hijack this system to determine if there is any structure we can exploit
# default: SVD is most efficient with BlockedArray
function eigencopy_oftype(A::AbstractBlockArray, S)
  return BlockedMatrix{S}(A)
end

function svd!(A::BlockedMatrix; full::Bool=false, alg::Algorithm=default_svd_alg(A))
  F = svd!(parent(A); full, alg)

  # restore block pattern
  m = length(F.S)
  bax1, bax2, bax3 = axes(A, 1), blockedrange([m]), axes(A, 2)

  u = BlockedArray(F.U, (bax1, bax2))
  s = BlockedVector(F.S, (bax2,))
  vt = BlockedArray(F.Vt, (bax2, bax3))
  return SVD(u, s, vt)
end
