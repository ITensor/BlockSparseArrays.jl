module BlockSparseArraysTensorAlgebraExt
using BlockArrays: AbstractBlockedUnitRange

using TensorAlgebra: TensorAlgebra, FusionStyle, BlockReshapeFusion
using TensorProducts: OneToOne

using BlockSparseArrays: AbstractBlockSparseArray, blockreshape

TensorAlgebra.FusionStyle(::AbstractBlockedUnitRange) = BlockReshapeFusion()

function TensorAlgebra.fusedims(
  ::BlockReshapeFusion, a::AbstractArray, axes::AbstractUnitRange...
)
  return blockreshape(a, axes)
end

function TensorAlgebra.splitdims(
  ::BlockReshapeFusion, a::AbstractArray, axes::AbstractUnitRange...
)
  return blockreshape(a, axes)
end

using BlockArrays:
  AbstractBlockVector,
  AbstractBlockedUnitRange,
  Block,
  BlockIndexRange,
  blockedrange,
  blocks
using BlockSparseArrays:
  BlockSparseArrays,
  AbstractBlockSparseArray,
  AbstractBlockSparseArrayInterface,
  AbstractBlockSparseMatrix,
  BlockSparseArray,
  BlockSparseArrayInterface,
  BlockSparseMatrix,
  BlockSparseVector,
  block_merge
using DerivableInterfaces: @interface
using GradedUnitRanges:
  GradedUnitRanges,
  AbstractGradedUnitRange,
  blockmergesortperm,
  blocksortperm,
  dual,
  invblockperm,
  nondual,
  unmerged_tensor_product
using LinearAlgebra: Adjoint, Transpose
using TensorAlgebra:
  TensorAlgebra, FusionStyle, BlockReshapeFusion, SectorFusion, fusedims, splitdims

# TODO: Make a `ReduceWhile` library.
include("reducewhile.jl")

TensorAlgebra.FusionStyle(::AbstractGradedUnitRange) = SectorFusion()

# TODO: Need to implement this! Will require implementing
# `block_merge(a::AbstractUnitRange, blockmerger::BlockedUnitRange)`.
function BlockSparseArrays.block_merge(
  a::AbstractGradedUnitRange, blockmerger::AbstractBlockedUnitRange
)
  return a
end

# Sort the blocks by sector and then merge the common sectors.
function block_mergesort(a::AbstractArray)
  I = blockmergesortperm.(axes(a))
  return a[I...]
end

function TensorAlgebra.fusedims(
  ::SectorFusion, a::AbstractArray, merged_axes::AbstractUnitRange...
)
  # First perform a fusion using a block reshape.
  # TODO avoid groupreducewhile. Require refactor of fusedims.
  unmerged_axes = groupreducewhile(
    unmerged_tensor_product, axes(a), length(merged_axes); init=OneToOne()
  ) do i, axis
    return length(axis) ≤ length(merged_axes[i])
  end

  a_reshaped = fusedims(BlockReshapeFusion(), a, unmerged_axes...)
  # Sort the blocks by sector and merge the equivalent sectors.
  return block_mergesort(a_reshaped)
end

function TensorAlgebra.splitdims(
  ::SectorFusion, a::AbstractArray, split_axes::AbstractUnitRange...
)
  # First, fuse axes to get `blockmergesortperm`.
  # Then unpermute the blocks.
  axes_prod = groupreducewhile(
    unmerged_tensor_product, split_axes, ndims(a); init=OneToOne()
  ) do i, axis
    return length(axis) ≤ length(axes(a, i))
  end
  blockperms = blocksortperm.(axes_prod)
  sorted_axes = map((r, I) -> only(axes(r[I])), axes_prod, blockperms)

  # TODO: This is doing extra copies of the blocks,
  # use `@view a[axes_prod...]` instead.
  # That will require implementing some reindexing logic
  # for this combination of slicing.
  a_unblocked = a[sorted_axes...]
  a_blockpermed = a_unblocked[invblockperm.(blockperms)...]
  return splitdims(BlockReshapeFusion(), a_blockpermed, split_axes...)
end

# TODO: Handle this through some kind of trait dispatch, maybe
# a `SymmetryStyle`-like trait to check if the block sparse
# matrix has graded axes.
function Base.axes(a::Adjoint{<:Any,<:AbstractBlockSparseMatrix})
  return dual.(reverse(axes(a')))
end

end
