using MatrixAlgebraKit:
  MatrixAlgebraKit,
  TruncatedAlgorithm,
  TruncationStrategy,
  diagview,
  eig_trunc!,
  eigh_trunc!,
  findtruncated,
  svd_trunc!,
  truncate!

"""
    BlockDiagonalTruncationStrategy(strategy::TruncationStrategy)

A wrapper for `TruncationStrategy` that implements the wrapped strategy on a block-by-block
basis, which is possible if the input matrix is a block-diagonal matrix.
"""
struct BlockDiagonalTruncationStrategy{T<:TruncationStrategy} <: TruncationStrategy
  strategy::T
end

function BlockDiagonalTruncationStrategy(alg::BlockPermutedDiagonalAlgorithm)
  return BlockDiagonalTruncationStrategy(alg.strategy)
end

function MatrixAlgebraKit.svd_trunc!(
  A::AbstractBlockSparseMatrix,
  out,
  alg::TruncatedAlgorithm{<:BlockPermutedDiagonalAlgorithm},
)
  Ad, (invrowperm, invcolperm) = blockdiagonalize(A)
  blockalg = BlockDiagonalAlgorithm(alg.alg)
  blockstrategy = BlockDiagonalTruncationStrategy(alg.trunc)
  Ud, S, Vᴴd = svd_trunc!(Ad, TruncatedAlgorithm(blockalg, blockstrategy))

  U = transform_rows(Ud, invrowperm)
  Vᴴ = transform_cols(Vᴴd, invcolperm)

  return U, S, Vᴴ
end

for f in [:eig_trunc!, :eigh_trunc!]
  @eval begin
    function MatrixAlgebraKit.truncate!(
      ::typeof($f),
      (D, V)::NTuple{2,AbstractBlockSparseMatrix},
      strategy::TruncationStrategy,
    )
      return truncate!($f, (D, V), BlockDiagonalTruncationStrategy(strategy))
    end
  end
end

# cannot use regular slicing here: I want to slice without altering blockstructure
# solution: use boolean indexing and slice the mask, effectively cheaply inverting the map
function MatrixAlgebraKit.findtruncated(
  values::AbstractVector, strategy::BlockDiagonalTruncationStrategy
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
  strategy::BlockDiagonalTruncationStrategy,
)
  I = findtruncated(diag(S), strategy)
  return (U[:, I], S[I, I], Vᴴ[I, :])
end
for f in [:eig_trunc!, :eigh_trunc!]
  @eval begin
    function MatrixAlgebraKit.truncate!(
      ::typeof($f),
      (D, V)::NTuple{2,AbstractBlockSparseMatrix},
      strategy::BlockDiagonalTruncationStrategy,
    )
      I = findtruncated(diag(D), strategy)
      return (D[I, I], V[:, I])
    end
  end
end
