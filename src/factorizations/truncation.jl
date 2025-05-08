using MatrixAlgebraKit: TruncationStrategy, diagview, svd_trunc!

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
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::TBlockUSVᴴ,
  strategy::BlockPermutedDiagonalTruncationStrategy,
)
  ind = MatrixAlgebraKit.findtruncated(diagview(S), strategy.strategy)
  # cannot use regular slicing here: I want to slice without altering blockstructure
  # solution: use boolean indexing and slice the mask, effectively cheaply inverting the map
  indexmask = falses(size(S, 1))
  indexmask[ind] .= true

  # first determine the block structure of the output to avoid having assumptions on the
  # data structures
  ax = axes(S, 1)
  counter = Base.Fix1(count, Base.Fix1(getindex, indexmask))
  Slengths = filter!(>(0), map(counter, blocks(ax)))
  Sax = blockedrange(Slengths)
  Ũ = similar(U, axes(U, 1), Sax)
  S̃ = similar(S, Sax, Sax)
  Ṽᴴ = similar(Vᴴ, Sax, axes(Vᴴ, 2))

  # then loop over the blocks and assign the data
  # TODO: figure out if we can presort and loop over the blocks -
  # for now this has issues with missing blocks
  bI_Us = collect(eachblockstoredindex(U))
  bI_Ss = collect(eachblockstoredindex(S))
  bI_Vᴴs = collect(eachblockstoredindex(Vᴴ))

  I′ = 0 # number of skipped blocks that got fully truncated
  for (I, b) in enumerate(blocks(ax))
    mask = indexmask[b]

    if !any(mask)
      I′ += 1
      continue
    end

    bU_id = @something findfirst(x -> last(Tuple(x)) == Block(I), bI_Us) error(
      "No U-block found for $I"
    )
    bU = Tuple(bI_Us[bU_id])
    Ũ[bU[1], bU[2] - Block(I′)] = view(U, bU...)[:, mask]

    bVᴴ_id = @something findfirst(x -> first(Tuple(x)) == Block(I), bI_Vᴴs) error(
      "No Vᴴ-block found for $I"
    )
    bVᴴ = Tuple(bI_Vᴴs[bVᴴ_id])
    Ṽᴴ[bVᴴ[1] - Block(I′), bVᴴ[2]] = view(Vᴴ, bVᴴ...)[mask, :]

    bS_id = @something findfirst(x -> last(Tuple(x)) == Block(I), bI_Ss) error(
      "No S-block found for $I"
    )
    bS = Tuple(bI_Ss[bS_id])
    S̃[(bS .- Block(I′))...] = Diagonal(diagview(view(S, bS...))[mask])
  end

  return Ũ, S̃, Ṽᴴ
end

