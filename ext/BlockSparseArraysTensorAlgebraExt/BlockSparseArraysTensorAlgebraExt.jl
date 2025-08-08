module BlockSparseArraysTensorAlgebraExt

using BlockSparseArrays: AbstractBlockSparseArray, blockreshape
using TensorAlgebra:
  TensorAlgebra,
  BlockedTrivialPermutation,
  BlockedTuple,
  FusionStyle,
  ReshapeFusion,
  fuseaxes

struct BlockReshapeFusion <: FusionStyle end

function TensorAlgebra.FusionStyle(::Type{<:AbstractBlockSparseArray})
  return BlockReshapeFusion()
end

using BlockArrays: Block, blocklength, blocks
using BlockSparseArrays: blocksparse
using SparseArraysBase: eachstoredindex
using TensorAlgebra: TensorAlgebra, matricize, unmatricize
function TensorAlgebra.matricize(
  ::BlockReshapeFusion, a::AbstractArray, biperm::BlockedTrivialPermutation{2}
)
  ax = fuseaxes(axes(a), biperm)
  reshaped_blocks_a = reshape(blocks(a), map(blocklength, ax))
  bs = Dict(
    Block(Tuple(I)) => matricize(reshaped_blocks_a[I], biperm) for
    I in eachstoredindex(reshaped_blocks_a)
  )
  return blocksparse(bs, ax)
end

using BlockArrays: blocklengths
function TensorAlgebra.unmatricize(
  ::BlockReshapeFusion,
  m::AbstractMatrix,
  blocked_ax::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  ax = Tuple(blocked_ax)
  reshaped_blocks_m = reshape(blocks(m), map(blocklength, ax))
  function f(I)
    block_axes_I = BlockedTuple(
      map(ntuple(identity, length(ax))) do i
        return Base.axes1(ax[i][Block(I[i])])
      end,
      blocklengths(blocked_ax),
    )
    return unmatricize(reshaped_blocks_m[I], block_axes_I)
  end
  bs = Dict(Block(Tuple(I)) => f(I) for I in eachstoredindex(reshaped_blocks_m))
  return blocksparse(bs, ax)
end

end
