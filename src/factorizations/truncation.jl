using MatrixAlgebraKit: TruncationStrategy, diagview, svd_trunc!

function MatrixAlgebraKit.diagview(A::BlockSparseMatrix{T,Diagonal{T,Vector{T}}}) where {T}
  D = BlockSparseVector{T}(undef, axes(A, 1))
  for I in eachblockstoredindex(A)
    if ==(Int.(Tuple(I))...)
      D[Tuple(I)[1]] = diagview(A[I])
    end
  end
  return D
end

"""
    BlockPermutedDiagonalTruncationStrategy(strategy::TruncationStrategy)

A wrapper for `TruncationStrategy` that implements the wrapped strategy on a block-by-block
basis, which is possible if the input matrix is a block-diagonal matrix or a block permuted
block-diagonal matrix.
"""
struct BlockPermutedDiagonalTruncationStrategy{T<:TruncationStrategy} <: TruncationStrategy
  strategy::T
end

const TBlockUSVᴴ = Tuple{
  <:AbstractBlockSparseMatrix,<:AbstractBlockSparseMatrix,<:AbstractBlockSparseMatrix
}

function MatrixAlgebraKit.truncate!(
  ::typeof(svd_trunc!), (U, S, Vᴴ)::TBlockUSVᴴ, strategy::TruncationStrategy
)
  # TODO assert blockdiagonal
  return MatrixAlgebraKit.truncate!(
    svd_trunc!, (U, S, Vᴴ), BlockPermutedDiagonalTruncationStrategy(strategy)
  )
end

# cannot use regular slicing here: I want to slice without altering blockstructure
# solution: use boolean indexing and slice the mask, effectively cheaply inverting the map
function MatrixAlgebraKit.findtruncated(
  values::AbstractVector, strategy::BlockPermutedDiagonalTruncationStrategy
)
  ind = MatrixAlgebraKit.findtruncated(values, strategy.strategy)
  indexmask = falses(length(values))
  indexmask[ind] .= true
  return indexmask
end

function MatrixAlgebraKit.truncate!(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::TBlockUSVᴴ,
  strategy::BlockPermutedDiagonalTruncationStrategy,
)
  I = MatrixAlgebraKit.findtruncated(diagview(S), strategy)
  return (U[:, I], S[I, I], Vᴴ[I, :])
end
