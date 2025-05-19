using BlockArrays: BlockRange, blockisequal
using DerivableInterfaces: @interface, AbstractArrayInterface, interface
using GPUArraysCore: @allowscalar

# Check if the block structures are the same.
function same_block_structure(as::AbstractArray...)
  isempty(as) && return true
  return all(
    ntuple(ndims(first(as))) do dim
      ax = map(Base.Fix2(axes, dim), as)
      return blockisequal(ax...)
    end,
  )
end

# Find the common stored blocks, assuming the block structures are the same.
function union_eachblockstoredindex(as::AbstractArray...)
  return ∪(map(eachblockstoredindex, as)...)
end

function map_blockwise!(f, a_dest::AbstractArray, a_srcs::AbstractArray...)
  # TODO: This assumes element types are numbers, generalize this logic.
  f_preserves_zeros = f(zero.(eltype.(a_srcs))...) == zero(eltype(a_dest))
  Is = if f_preserves_zeros
    union_eachblockstoredindex(a_dest, a_srcs...)
  else
    BlockRange(a_dest)
  end
  for I in Is
    # TODO: Use:
    # block_dest = @view a_dest[I]
    # or:
    # block_dest = @view! a_dest[I]
    block_dest = blocks_maybe_single(a_dest)[Int.(Tuple(I))...]
    # TODO: Use:
    # block_srcs = map(a_src -> @view(a_src[I]), a_srcs)
    block_srcs = map(a_srcs) do a_src
      return blocks_maybe_single(a_src)[Int.(Tuple(I))...]
    end
    # TODO: Use `map!!` to handle immutable blocks.
    map!(f, block_dest, block_srcs...)
    # Replace the entire block, handles initializing new blocks
    # or if blocks are immutable.
    # TODO: Use `a_dest[I] = block_dest`.
    blocks(a_dest)[Int.(Tuple(I))...] = block_dest
  end
  return a_dest
end

# TODO: Rewrite this so that it takes the blocking structure
# made by combining the blocking of the axes (i.e. the blocking that
# is used to determine `union_stored_blocked_cartesianindices(...)`).
# `reblock` is a partial solution to that, but a bit ad-hoc.
## TODO: Make this an `@interface AbstractBlockSparseArrayInterface` function.
@interface interface::AbstractBlockSparseArrayInterface function Base.map!(
  f, a_dest::AbstractArray, a_srcs::AbstractArray...
)
  if isempty(a_srcs)
    error("Can't call `map!` with zero source terms.")
  end
  if iszero(ndims(a_dest))
    @interface interface map_zero_dim!(f, a_dest, a_srcs...)
    return a_dest
  end
  if same_block_structure(a_dest, a_srcs...)
    map_blockwise!(f, a_dest, a_srcs...)
    return a_dest
  end
  # TODO: This assumes element types are numbers, generalize this logic.
  f_preserves_zeros = f(zero.(eltype.(a_srcs))...) == zero(eltype(a_dest))
  a_dest, a_srcs = reblock(a_dest), reblock.(a_srcs)
  for I in union_stored_blocked_cartesianindices(a_dest, a_srcs...)
    BI_dest = blockindexrange(a_dest, I)
    BI_srcs = map(a_src -> blockindexrange(a_src, I), a_srcs)
    # TODO: Investigate why this doesn't work:
    # block_dest = @view a_dest[_block(BI_dest)]
    block_dest = blocks_maybe_single(a_dest)[Int.(Tuple(_block(BI_dest)))...]
    # TODO: Investigate why this doesn't work:
    # block_srcs = ntuple(i -> @view(a_srcs[i][_block(BI_srcs[i])]), length(a_srcs))
    block_srcs = ntuple(length(a_srcs)) do i
      return blocks_maybe_single(a_srcs[i])[Int.(Tuple(_block(BI_srcs[i])))...]
    end
    subblock_dest = @view block_dest[BI_dest.indices...]
    subblock_srcs = ntuple(i -> @view(block_srcs[i][BI_srcs[i].indices...]), length(a_srcs))
    I_dest = CartesianIndex(Int.(Tuple(_block(BI_dest))))
    # If the function preserves zero values and all of the source blocks are zero,
    # the output block will be zero. In that case, if the block isn't stored yet,
    # don't do anything.
    if f_preserves_zeros && all(iszero, subblock_srcs) && !isstored(blocks(a_dest), I_dest)
      continue
    end
    # TODO: Use `map!!` to handle immutable blocks.
    map!(f, subblock_dest, subblock_srcs...)
    # Replace the entire block, handles initializing new blocks
    # or if blocks are immutable.
    blocks(a_dest)[I_dest] = block_dest
  end
  return a_dest
end

@interface ::AbstractBlockSparseArrayInterface function Base.mapreduce(
  f, op, as::AbstractArray...; kwargs...
)
  # TODO: Define an `init` value based on the element type.
  return @interface interface(blocks.(as)...) mapreduce(
    block -> mapreduce(f, op, block), op, blocks.(as)...; kwargs...
  )
end

@interface ::AbstractBlockSparseArrayInterface function Base.iszero(a::AbstractArray)
  # TODO: Just call `iszero(blocks(a))`?
  return @interface interface(blocks(a)) iszero(blocks(a))
end

@interface ::AbstractBlockSparseArrayInterface function Base.isreal(a::AbstractArray)
  # TODO: Just call `isreal(blocks(a))`?
  return @interface interface(blocks(a)) isreal(blocks(a))
end

# Helper functions for block sparse map.

# Returns `Vector{<:CartesianIndices}`
function union_stored_blocked_cartesianindices(as::Vararg{AbstractArray})
  combined_axes = combine_axes(axes.(as)...)
  stored_blocked_cartesianindices_as = map(as) do a
    return blocked_cartesianindices(axes(a), combined_axes, eachblockstoredindex(a))
  end
  return ∪(stored_blocked_cartesianindices_as...)
end

# This is used by `map` to get the output axes.
# This is type piracy, try to avoid this, maybe requires defining `map`.
## Base.promote_shape(a1::Tuple{Vararg{BlockedUnitRange}}, a2::Tuple{Vararg{BlockedUnitRange}}) = combine_axes(a1, a2)

reblock(a) = a

# `map!` specialized to zero-dimensional inputs.
function map_zero_dim! end

@interface ::AbstractArrayInterface function map_zero_dim!(
  f, a_dest::AbstractArray, a_srcs::AbstractArray...
)
  @allowscalar a_dest[] = f.(map(a_src -> a_src[], a_srcs)...)
  return a_dest
end

# TODO: Do we need this function or can we just use `map`?
# Probably it should be a special version of `map` where we
# specify the function preserves zeros, i.e.
# `map(f, a; preserves_zeros=true)` or `@preserves_zeros map(f, a)`.
function map_stored_blocks(f, a::AbstractArray)
  block_stored_indices = collect(eachblockstoredindex(a))
  if isempty(block_stored_indices)
    blocktype_a′ = Base.promote_op(f, blocktype(a))
    return BlockSparseArray{eltype(blocktype_a′),ndims(a),blocktype_a′}(undef, axes(a))
  end
  stored_blocks = map(B -> f(@view!(a[B])), block_stored_indices)
  blocktype_a′ = eltype(stored_blocks)
  a′ = BlockSparseArray{eltype(blocktype_a′),ndims(a),blocktype_a′}(undef, axes(a))
  for (B, b) in zip(block_stored_indices, stored_blocks)
    a′[B] = b
  end
  return a′
end
