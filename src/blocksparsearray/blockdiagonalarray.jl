# type alias for block-diagonal
using LinearAlgebra: Diagonal

const BlockDiagonal{T,A,Axes,V<:AbstractVector{A}} = BlockSparseMatrix{
  T,A,Diagonal{A,V},Axes
}
const BlockSparseDiagonal{T,A<:AbstractBlockSparseVector{T}} = Diagonal{T,A}

@interface interface::BlockSparseArrayInterface function blocks(a::BlockSparseDiagonal)
  return Diagonal(Diagonal.(blocks(a.diag)))
end

function BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix})
  return BlockSparseArray(
    Diagonal(blocks), (blockedrange(size.(blocks, 1)), blockedrange(size.(blocks, 2)))
  )
end
