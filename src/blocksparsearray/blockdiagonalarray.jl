# type alias for block-diagonal
using LinearAlgebra: Diagonal

const BlockDiagonal{T,A,Axes,V<:AbstractVector{A}} = BlockSparseMatrix{
  T,A,Diagonal{A,V},Axes
}

function BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix})
  return BlockSparseArray(
    Diagonal(blocks), (blockedrange(size.(blocks, 1)), blockedrange(size.(blocks, 2)))
  )
end

# SVD implementation
function eigencopy_oftype(A::BlockDiagonal, T)
  diag = map(Base.Fix2(eigencopy_oftype, T), A.blocks.diag)
  return BlockDiagonal(diag)
end

function svd(A::BlockDiagonal; kwargs...)
  return svd!(eigencopy_oftype(A, LinearAlgebra.eigtype(eltype(A))); kwargs...)
end
function svd!(A::BlockDiagonal; full::Bool=false, alg::Algorithm=default_svd_alg(A))
  # TODO: handle full
  F = map(a -> svd!(a; full, alg), blocks(A).diag)
  Us = map(Base.Fix2(getproperty, :U), F)
  Ss = map(Base.Fix2(getproperty, :S), F)
  Vts = map(Base.Fix2(getproperty, :Vt), F)
  return SVD(BlockDiagonal(Us), mortar(Ss), BlockDiagonal(Vts))
end
