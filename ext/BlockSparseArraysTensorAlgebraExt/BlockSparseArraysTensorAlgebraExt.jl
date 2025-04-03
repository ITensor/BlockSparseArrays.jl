module BlockSparseArraysTensorAlgebraExt

using BlockArrays: AbstractBlockedUnitRange
using BlockSparseArrays: AbstractBlockSparseArray, blockreshape
using TensorAlgebra: TensorAlgebra, FusionStyle, BlockReshapeFusion

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

end
