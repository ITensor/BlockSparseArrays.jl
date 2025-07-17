using BlockArrays:
  BlockArrays,
  AbstractBlockedUnitRange,
  AbstractBlockVector,
  Block,
  BlockIndex,
  BlockIndexRange,
  BlockRange,
  BlockSlice,
  BlockVector,
  block,
  blockedrange,
  blockfirsts,
  blockindex,
  blocklengths,
  findblock,
  findblockindex,
  mortar

# Get the axes of each block of a block array.
function eachblockaxes(a::AbstractArray)
  return map(axes, blocks(a))
end

axis(a::AbstractVector) = axes(a, 1)

# Get the axis of each block of a blocked unit
# range.
function eachblockaxis(a::AbstractVector)
  return map(axis, blocks(a))
end
function blockaxistype(a::AbstractVector)
  return eltype(eachblockaxis(a))
end

# Take a collection of axes and mortar them
# into a single blocked axis.
function mortar_axis(axs)
  return blockrange(axs)
end
function mortar_axis(axs::Vector{<:Base.OneTo{<:Integer}})
  return blockedrange(length.(axs))
end

# Custom `BlockedUnitRange` constructor that takes a unit range
# and a set of block lengths, similar to `BlockArray(::AbstractArray, blocklengths...)`.
function blockedunitrange(a::AbstractUnitRange, blocklengths)
  blocklengths_shifted = copy(blocklengths)
  blocklengths_shifted[1] += (first(a) - 1)
  blocklasts = cumsum(blocklengths_shifted)
  return BlockArrays._BlockedUnitRange(first(a), blocklasts)
end

# TODO: Move this to a `BlockArraysExtensions` library.
# TODO: Rename this. `BlockArrays.findblock(a, k)` finds the
# block of the value `k`, while this finds the block of the index `k`.
# This could make use of the `BlockIndices` object, i.e. `block(BlockIndices(a)[index])`.
function blockedunitrange_findblock(a::AbstractBlockedUnitRange, index::Integer)
  @boundscheck index in 1:length(a) || throw(BoundsError(a, index))
  return @inbounds findblock(a, index + first(a) - 1)
end

# TODO: Move this to a `BlockArraysExtensions` library.
# TODO: Rename this. `BlockArrays.findblockindex(a, k)` finds the
# block index of the value `k`, while this finds the block index of the index `k`.
# This could make use of the `BlockIndices` object, i.e. `BlockIndices(a)[index]`.
function blockedunitrange_findblockindex(a::AbstractBlockedUnitRange, index::Integer)
  @boundscheck index in 1:length(a) || throw(BoundsError())
  return @inbounds findblockindex(a, index + first(a) - 1)
end

function blockedunitrange_getindices(a::AbstractUnitRange, indices)
  return a[indices]
end

# TODO: Move this to a `BlockArraysExtensions` library.
# Like `a[indices]` but preserves block structure.
# TODO: Consider calling this something else, for example
# `blocked_getindex`. See the discussion here:
# https://github.com/JuliaArrays/BlockArrays.jl/issues/347
function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  first_blockindex = blockedunitrange_findblockindex(a, first(indices))
  last_blockindex = blockedunitrange_findblockindex(a, last(indices))
  first_block = block(first_blockindex)
  last_block = block(last_blockindex)
  blocklengths = if first_block == last_block
    [length(indices)]
  else
    map(first_block:last_block) do block
      if block == first_block
        return length(a[first_block]) - blockindex(first_blockindex) + 1
      end
      if block == last_block
        return blockindex(last_blockindex)
      end
      return length(a[block])
    end
  end
  return blockedunitrange(indices .+ (first(a) - 1), blocklengths)
end

# TODO: Make sure this handles block labels (AbstractGradedUnitRange) correctly.
# TODO: Make a special case for `BlockedVector{<:Block{1},<:BlockRange{1}}`?
# For example:
# ```julia
# blocklengths = map(bs -> sum(b -> length(a[b]), bs), blocks(indices))
# return blockedrange(blocklengths)
# ```
function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange, indices::AbstractBlockVector{<:Block{1}}
)
  blks = map(bs -> mortar(map(b -> a[b], bs)), blocks(indices))
  # We pass `length.(blks)` to `mortar` in order
  # to pass block labels to the axes of the output,
  # if they exist. This makes it so that
  # `only(axes(a[indices])) isa `GradedUnitRange`
  # if `a isa `GradedUnitRange`, for example.
  # Note there is a more specialized definition:
  # ```julia
  # function blockedunitrange_getindices(
  #   a::AbstractGradedUnitRange, indices::AbstractBlockVector{<:Block{1}}
  # )
  # ```
  # that does a better job of preserving labels, since `length`
  # may drop labels for certain block types.
  return mortar(blks, length.(blks))
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices::BlockIndexRange)
  return a[block(indices)][only(indices.indices)]
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices::BlockSlice)
  # TODO: Is this a good definition? It ignores `indices.indices`.
  return a[indices.block]
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange, indices::AbstractVector{<:Integer}
)
  return map(index -> a[index], indices)
end

# TODO: Move this to a `BlockArraysExtensions` library.
# TODO: Make a special definition for `BlockedVector{<:Block{1}}` in order
# to merge blocks.
function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange, indices::AbstractVector{<:Union{Block{1},BlockIndexRange{1}}}
)
  # Without converting `indices` to `Vector`,
  # mapping `indices` outputs a `BlockVector`
  # which is harder to reason about.
  blocks = map(index -> a[index], Vector(indices))
  # We pass `length.(blocks)` to `mortar` in order
  # to pass block labels to the axes of the output,
  # if they exist. This makes it so that
  # `only(axes(a[indices])) isa `GradedUnitRange`
  # if `a isa `GradedUnitRange`, for example.
  return mortar(blocks, length.(blocks))
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices::Block{1})
  return a[indices]
end

function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange,
  indices::BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}},
)
  return mortar(map(b -> a[b], blocks(indices)))
end

function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange, indices::AbstractVector{Bool}
)
  blocked_indices = BlockedVector(indices, axes(a))
  bs = map(Base.OneTo(blocklength(blocked_indices))) do b
    binds = blocked_indices[Block(b)]
    bstart = blockfirsts(only(axes(blocked_indices)))[b]
    return findall(binds) .+ (bstart - 1)
  end
  return mortar(filter(!isempty, bs))
end

# TODO: Move this to a `BlockArraysExtensions` library.
function blockedunitrange_getindices(a::AbstractBlockedUnitRange, indices)
  return error("Not implemented.")
end

# The blocks of the corresponding slice.
_blocks(a::AbstractUnitRange, indices) = error("Not implemented")
function _blocks(a::AbstractUnitRange, indices::AbstractUnitRange)
  return findblock(a, first(indices)):findblock(a, last(indices))
end
function _blocks(a::AbstractUnitRange, indices::BlockRange)
  return indices
end

# Slice `a` by `I`, returning a:
# `BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}}`
# with the `BlockIndex{1}` corresponding to each value of `I`.
function to_blockindices(a::AbstractBlockedUnitRange{<:Integer}, I::UnitRange{<:Integer})
  return mortar(
    map(blocks(blockedunitrange_getindices(a, I))) do r
      bi_first = findblockindex(a, first(r))
      bi_last = findblockindex(a, last(r))
      @assert block(bi_first) == block(bi_last)
      return block(bi_first)[blockindex(bi_first):blockindex(bi_last)]
    end,
  )
end

struct GenericBlockIndex{N,TI<:Tuple{Vararg{Integer,N}},Tα<:Tuple{Vararg{Any,N}}}
  I::TI
  α::Tα
end
@inline function GenericBlockIndex(a::NTuple{N,Block{1}}, b::Tuple) where {N}
  return GenericBlockIndex(Int.(a), b)
end
@inline function GenericBlockIndex(::Tuple{}, b::Tuple{})
  return GenericBlockIndex{0,Tuple{},Tuple{}}((), ())
end
@inline GenericBlockIndex(a::Integer, b) = GenericBlockIndex((a,), (b,))
@inline GenericBlockIndex(a::Tuple, b) = GenericBlockIndex(a, (b,))
@inline GenericBlockIndex(a::Integer, b::Tuple) = GenericBlockIndex((a,), b)
@inline GenericBlockIndex() = GenericBlockIndex((), ())
@inline GenericBlockIndex(a::Block, b::Tuple) = GenericBlockIndex(a.n, b)
@inline GenericBlockIndex(a::Block, b) = GenericBlockIndex(a, (b,))
@inline function GenericBlockIndex(
  I::Tuple{Vararg{Integer,N}}, α::Tuple{Vararg{Any,M}}
) where {M,N}
  M <= N || throw(ArgumentError("number of indices must not exceed the number of blocks"))
  α2 = ntuple(k -> k <= M ? α[k] : 1, N)
  GenericBlockIndex(I, α2)
end
BlockArrays.block(b::GenericBlockIndex) = Block(b.I...)
BlockArrays.blockindex(b::GenericBlockIndex{1}) = b.α[1]
function GenericBlockIndex(indcs::Tuple{Vararg{GenericBlockIndex{1},N}}) where {N}
  GenericBlockIndex(block.(indcs), blockindex.(indcs))
end

function Base.checkindex(
  ::Type{Bool}, axis::AbstractBlockedUnitRange, ind::GenericBlockIndex{1}
)
  return checkindex(Bool, axis, block(ind)) &&
         checkbounds(Bool, axis[block(ind)], blockindex(ind))
end
Base.to_index(i::GenericBlockIndex) = i

function print_tuple_elements(io::IO, @nospecialize(t))
  if !isempty(t)
    print(io, t[1])
    for n in t[2:end]
      print(io, ", ", n)
    end
  end
  return nothing
end
function Base.show(io::IO, B::GenericBlockIndex)
  show(io, Block(B.I...))
  print(io, "[")
  print_tuple_elements(io, B.α)
  print(io, "]")
  return nothing
end

# https://github.com/JuliaArrays/BlockArrays.jl/blob/v1.6.3/src/views.jl#L31-L32
_maybetail(::Tuple{}) = ()
_maybetail(t::Tuple) = Base.tail(t)
@inline function Base.to_indices(A, inds, I::Tuple{GenericBlockIndex{1},Vararg{Any}})
  return (inds[1][I[1]], to_indices(A, _maybetail(inds), Base.tail(I))...)
end

using Base: @propagate_inbounds
@propagate_inbounds function Base.getindex(b::AbstractVector, K::GenericBlockIndex{1})
  return b[Block(K.I[1])][K.α[1]]
end
@propagate_inbounds function Base.getindex(
  b::AbstractArray{T,N}, K::GenericBlockIndex{N}
) where {T,N}
  return b[block(K)][K.α...]
end
@propagate_inbounds function Base.getindex(
  b::AbstractArray, K::GenericBlockIndex{1}, J::GenericBlockIndex{1}...
)
  return b[GenericBlockIndex(tuple(K, J...))]
end

# TODO: Delete this once `BlockArrays.BlockIndex` is generalized.
@inline function Base.to_indices(
  A, inds, I::Tuple{AbstractVector{<:GenericBlockIndex{1}},Vararg{Any}}
)
  return (unblock(A, inds, I), to_indices(A, _maybetail(inds), Base.tail(I))...)
end

# This is a specialization of `BlockArrays.unblock`:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/v1.6.3/src/views.jl#L8-L11
# that is used in the `to_indices` logic for blockwise slicing in
# BlockArrays.jl.
# TODO: Ideally this would be defined in BlockArrays.jl once the slicing
# there is made more generic.
function BlockArrays.unblock(A, inds, I::Tuple{GenericBlockIndex{1},Vararg{Any}})
  B = first(I)
  return _blockslice(B, inds[1][B])
end

# Work around the fact that it is type piracy to define
# `Base.getindex(a::Block, b...)`.
_getindex(a::Block{N}, b::Vararg{Any,N}) where {N} = GenericBlockIndex(a, b)
_getindex(a::Block{N}, b::Vararg{Integer,N}) where {N} = a[b...]
_getindex(a::Block{N}, b::Vararg{AbstractUnitRange{<:Integer},N}) where {N} = a[b...]
_getindex(a::Block{N}, b::Vararg{AbstractVector,N}) where {N} = BlockIndexVector(a, b)
# Fix ambiguity.
_getindex(a::Block{0}) = a[]

struct BlockIndexVector{N,BT,I<:NTuple{N,AbstractVector},TB<:Integer} <: AbstractArray{BT,N}
  block::Block{N,TB}
  indices::I
  function BlockIndexVector{N,BT}(
    block::Block{N,TB}, indices::I
  ) where {N,BT,I<:NTuple{N,AbstractVector},TB<:Integer}
    return new{N,BT,I,TB}(block, indices)
  end
end
function BlockIndexVector{1,BT}(block::Block{1}, indices::AbstractVector) where {BT}
  return BlockIndexVector{1,BT}(block, (indices,))
end
function BlockIndexVector(
  block::Block{N,TB}, indices::NTuple{N,AbstractVector}
) where {N,TB<:Integer}
  BT = Base.promote_op(_getindex, typeof(block), eltype.(indices)...)
  return BlockIndexVector{N,BT}(block, indices)
end
function BlockIndexVector(block::Block{1}, indices::AbstractVector)
  return BlockIndexVector(block, (indices,))
end
Base.size(a::BlockIndexVector) = length.(a.indices)
function Base.getindex(a::BlockIndexVector{N}, I::Vararg{Integer,N}) where {N}
  return _getindex(Block(a), getindex.(a.indices, I)...)
end
BlockArrays.block(b::BlockIndexVector) = b.block
BlockArrays.Block(b::BlockIndexVector) = b.block

Base.copy(a::BlockIndexVector) = BlockIndexVector(a.block, copy.(a.indices))

# Copied from BlockArrays.BlockIndexRange.
function Base.show(io::IO, B::BlockIndexVector)
  show(io, Block(B))
  print(io, "[")
  print_tuple_elements(io, B.indices)
  print(io, "]")
end
Base.show(io::IO, ::MIME"text/plain", B::BlockIndexVector) = show(io, B)

function Base.getindex(b::AbstractBlockedUnitRange, Kkr::BlockIndexVector{1})
  return b[block(Kkr)][Kkr.indices...]
end

using ArrayLayouts: LayoutArray
@propagate_inbounds Base.getindex(b::AbstractArray{T,N}, K::BlockIndexVector{N}) where {T,N} = b[block(
  K
)][K.indices...]
@propagate_inbounds Base.getindex(b::LayoutArray{T,N}, K::BlockIndexVector{N}) where {T,N} = b[block(
  K
)][K.indices...]
@propagate_inbounds Base.getindex(b::LayoutArray{T,1}, K::BlockIndexVector{1}) where {T} = b[block(
  K
)][K.indices...]

function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange,
  indices::BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexVector{1}}},
)
  blks = map(b -> a[b], blocks(indices))
  # Preserve any extra structure in the axes, like a
  # Kronecker structure, symmetry sectors, etc.
  ax = mortar_axis(map(b -> axis(a[b]), blocks(indices)))
  return mortar(blks, (ax,))
end
function blockedunitrange_getindices(
  a::AbstractBlockedUnitRange,
  indices::BlockVector{<:GenericBlockIndex{1},<:Vector{<:BlockIndexVector{1}}},
)
  blks = map(b -> a[b], blocks(indices))
  # Preserve any extra structure in the axes, like a
  # Kronecker structure, symmetry sectors, etc.
  ax = mortar_axis(map(b -> axis(a[b]), blocks(indices)))
  return mortar(blks, (ax,))
end

# This is a specialization of `BlockArrays.unblock`:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/v1.6.3/src/views.jl#L8-L11
# that is used in the `to_indices` logic for blockwise slicing in
# BlockArrays.jl.
# TODO: Ideally this would be defined in BlockArrays.jl once the slicing
# there is made more generic.
function BlockArrays.unblock(A, inds, I::Tuple{BlockIndexVector{1},Vararg{Any}})
  B = first(I)
  return _blockslice(B, inds[1][B])
end

function to_blockindices(a::AbstractBlockedUnitRange{<:Integer}, I::AbstractArray{Bool})
  I_blocks = blocks(BlockedVector(I, blocklengths(a)))
  I′_blocks = map(eachindex(I_blocks)) do b
    I_b = findall(I_blocks[b])
    return BlockIndexVector(Block(b), I_b)
  end
  return mortar(filter(!isempty, I′_blocks))
end

# This handles non-blocked slices.
# For example:
# a = BlockSparseArray{Float64}([2, 2, 2, 2])
# I = BlockedVector(Block.(1:4), [2, 2])
# @views a[I][Block(1)]
to_blockindices(a::Base.OneTo{<:Integer}, I::UnitRange{<:Integer}) = I
