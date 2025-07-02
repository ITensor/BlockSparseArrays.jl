using MatrixAlgebraKit:
  TruncationStrategy,
  diagview,
  eig_trunc!,
  eigh_trunc!,
  findtruncated,
  svd_trunc!,
  truncate!

"""
    BlockPermutedDiagonalTruncationStrategy(strategy::TruncationStrategy)

A wrapper for `TruncationStrategy` that implements the wrapped strategy on a block-by-block
basis, which is possible if the input matrix is a block-diagonal matrix or a block permuted
block-diagonal matrix.
"""
struct BlockPermutedDiagonalTruncationStrategy{T<:TruncationStrategy} <: TruncationStrategy
  strategy::T
end

function MatrixAlgebraKit.truncate!(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::NTuple{3,AbstractBlockSparseMatrix},
  strategy::TruncationStrategy,
)
  # TODO assert blockdiagonal
  return truncate!(
    svd_trunc!, (U, S, Vᴴ), BlockPermutedDiagonalTruncationStrategy(strategy)
  )
end
for f in [:eig_trunc!, :eigh_trunc!]
  @eval begin
    function MatrixAlgebraKit.truncate!(
      ::typeof($f),
      (D, V)::NTuple{2,AbstractBlockSparseMatrix},
      strategy::TruncationStrategy,
    )
      return truncate!($f, (D, V), BlockPermutedDiagonalTruncationStrategy(strategy))
    end
  end
end

# cannot use regular slicing here: I want to slice without altering blockstructure
# solution: use boolean indexing and slice the mask, effectively cheaply inverting the map
function MatrixAlgebraKit.findtruncated(
  values::AbstractVector, strategy::BlockPermutedDiagonalTruncationStrategy
)
  ind = findtruncated(Vector(values), strategy.strategy)
  indexmask = falses(length(values))
  indexmask[ind] .= true
  return to_truncated_indices(values, indexmask)
end

# Allow customizing the indices output by `findtruncated`
# based on the type of `values`, for example to preserve
# a block or Kronecker structure.
to_truncated_indices(values, I) = I
function to_truncated_indices(values::AbstractBlockVector, I::AbstractVector{Bool})
  I′ = BlockedVector(I, blocklengths(axis(values)))
  blocks = map(BlockRange(values)) do b
    return _getindex(b, to_truncated_indices(values[b], I′[b]))
  end
  return blocks
end

function MatrixAlgebraKit.truncate!(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::NTuple{3,AbstractBlockSparseMatrix},
  strategy::BlockPermutedDiagonalTruncationStrategy,
)
  I = findtruncated(diag(S), strategy)
  return (U[:, I], S[I, I], Vᴴ[I, :])
end
for f in [:eig_trunc!, :eigh_trunc!]
  @eval begin
    function MatrixAlgebraKit.truncate!(
      ::typeof($f),
      (D, V)::NTuple{2,AbstractBlockSparseMatrix},
      strategy::BlockPermutedDiagonalTruncationStrategy,
    )
      I = findtruncated(diag(D), strategy)
      return (D[I, I], V[:, I])
    end
  end
end
