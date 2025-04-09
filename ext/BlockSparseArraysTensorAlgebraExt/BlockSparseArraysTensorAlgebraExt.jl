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

function TensorAlgebra.matricize(
  ::BlockReshapeFusion, a::AbstractArray, biperm::BlockedTrivialPermutation{2}
)
  new_axes = fuseaxes(axes(a), biperm)
  return blockreshape(a, new_axes)
end

function TensorAlgebra.unmatricize(
  ::BlockReshapeFusion,
  m::AbstractMatrix,
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  return blockreshape(m, Tuple(blocked_axes)...)
end

end
