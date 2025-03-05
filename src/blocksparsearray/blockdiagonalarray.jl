using BlockArrays: blockedrange
using DiagonalArrays: DiagonalArrays, diagonal
using LinearAlgebra: Diagonal

# type alias for block-diagonal
const BlockDiagonal{T,A,Axes,V<:AbstractVector{A}} = BlockSparseMatrix{
  T,A,Diagonal{A,V},Axes
}
const BlockSparseDiagonal{T,A<:AbstractBlockSparseVector{T}} = Diagonal{T,A}

@interface interface::BlockSparseArrayInterface function blocks(a::BlockSparseDiagonal)
  return Diagonal(Diagonal.(blocks(a.diag)))
end

function BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix})
  return _BlockSparseArray(
    Diagonal(blocks), blockedrange.((size.(blocks, 1), size.(blocks, 2)))
  )
end

function DiagonalArrays.diagonal(S::BlockSparseVector)
  D = similar(S, (axes(S, 1), axes(S, 1)))
  for bI in eachblockstoredindex(S)
    D[bI, bI] = diagonal(@view!(S[bI]))
  end
  return D
end
