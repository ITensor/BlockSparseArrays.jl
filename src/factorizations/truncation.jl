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

function similar_truncate(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::TBlockUSVᴴ,
  strategy::BlockPermutedDiagonalTruncationStrategy,
  indexmask=MatrixAlgebraKit.findtruncated(diagview(S), strategy),
)
  ax = axes(S, 1)
  counter = Base.Fix1(count, Base.Fix1(getindex, indexmask))
  s_lengths = filter!(>(0), map(counter, blocks(ax)))
  s_axis = blockedrange(s_lengths)
  Ũ = similar(U, axes(U, 1), s_axis)
  S̃ = similar(S, s_axis, s_axis)
  Ṽᴴ = similar(Vᴴ, s_axis, axes(Vᴴ, 2))
  return Ũ, S̃, Ṽᴴ
end

function MatrixAlgebraKit.truncate!(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::TBlockUSVᴴ,
  strategy::BlockPermutedDiagonalTruncationStrategy,
)
  indexmask = MatrixAlgebraKit.findtruncated(diagview(S), strategy)

  # first determine the block structure of the output to avoid having assumptions on the
  # data structures
  Ũ, S̃, Ṽᴴ = similar_truncate(svd_trunc!, (U, S, Vᴴ), strategy, indexmask)

  # then loop over the blocks and assign the data
  # TODO: figure out if we can presort and loop over the blocks -
  # for now this has issues with missing blocks
  bI_Us = collect(eachblockstoredindex(U))
  bI_Ss = collect(eachblockstoredindex(S))
  bI_Vᴴs = collect(eachblockstoredindex(Vᴴ))

  I′ = 0 # number of skipped blocks that got fully truncated
  ax = axes(S, 1)
  for I in 1:blocksize(ax, 1)
    b = ax[Block(I)]
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

    bS_id = findfirst(x -> last(Tuple(x)) == Block(I), bI_Ss)
    if !isnothing(bS_id)
      bS = Tuple(bI_Ss[bS_id])
      S̃[(bS .- Block(I′))...] = Diagonal(diagview(view(S, bS...))[mask])
    end
  end

  return Ũ, S̃, Ṽᴴ
end
