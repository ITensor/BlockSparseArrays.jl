module BlockSparseArraysTensorAlgebraExt

using BlockSparseArrays: AbstractBlockSparseArray, blockreshape
using TensorAlgebra:
  TensorAlgebra,
  AbstractBlockPermutation,
  BlockedTuple,
  FusionStyle,
  ReshapeFusion,
  fuseaxes

struct BlockReshapeFusion <: FusionStyle end

function TensorAlgebra.FusionStyle(::AbstractBlockSparseArray, ::ReshapeFusion)
  return BlockReshapeFusion()
end

function TensorAlgebra.matricize(
  ::BlockReshapeFusion, a::AbstractArray, biperm::AbstractBlockPermutation{2}
)
  a_perm = permutedims(a, Tuple(biperm))
  new_axes = fuseaxes(axes(a_perm), biperm)
  return blockreshape(a_perm, new_axes)
end

function TensorAlgebra.unmatricize(
  ::BlockReshapeFusion,
  m::AbstractMatrix,
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  return blockreshape(m, Tuple(blocked_axes)...)
end

end
