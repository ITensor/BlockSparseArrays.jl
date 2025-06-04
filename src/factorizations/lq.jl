using MatrixAlgebraKit: MatrixAlgebraKit, default_lq_algorithm, lq_compact!, lq_full!

# TODO: Delete once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/32 is merged.
function MatrixAlgebraKit.default_lq_algorithm(A::AbstractBlockSparseMatrix; kwargs...)
  return default_lq_algorithm(typeof(A); kwargs...)
end

# TODO: Delete once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/32 is merged.
function MatrixAlgebraKit.default_algorithm(
  ::typeof(lq_compact!), A::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  return default_lq_algorithm(A; kwargs...)
end

# TODO: Delete once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/32 is merged.
function MatrixAlgebraKit.default_algorithm(
  ::typeof(lq_full!), A::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  return default_lq_algorithm(A; kwargs...)
end

function MatrixAlgebraKit.default_lq_algorithm(
  A::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  alg = default_lq_algorithm(blocktype(A); kwargs...)
  return BlockPermutedDiagonalAlgorithm(alg)
end

function similar_output(
  ::typeof(lq_compact!), A, L_axis, alg::MatrixAlgebraKit.AbstractAlgorithm
)
  L = similar(A, axes(A, 1), L_axis)
  Q = similar(A, L_axis, axes(A, 2))
  return L, Q
end

function similar_output(
  ::typeof(lq_full!), A, L_axis, alg::MatrixAlgebraKit.AbstractAlgorithm
)
  L = similar(A, axes(A, 1), L_axis)
  Q = similar(A, L_axis, axes(A, 2))
  return L, Q
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(lq_compact!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)
  bmn = min(bm, bn)

  brows = eachblockaxis(axes(A, 1))
  bcols = eachblockaxis(axes(A, 2))
  l_axes = similar(brows, bmn)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    len = minimum(length, (brows[row], bcols[col]))
    l_axes[row] = bcols[col][Base.OneTo(len)]
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    len = minimum(length, (brows[row], bcols[col]))
    l_axes[row] = bcols[col][Base.OneTo(len)]
  end

  l_axis = mortar_axis(l_axes)
  L, Q = similar_output(lq_compact!, A, l_axis, alg)

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    L[brow, brow], Q[brow, bcol] = MatrixAlgebraKit.initialize_output(
      lq_compact!, @view!(A[bI]), alg.alg
    )
  end

  # allocate output for blocks that aren't present -- do we also fill identities here?
  for (row, col) in zip(emptyrows, emptycols)
    @view!(Q[Block(row, col)])
  end

  return L, Q
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(lq_full!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)

  bcols = eachblockaxis(axes(A, 2))
  l_axes = copy(bcols)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    l_axes[row] = bcols[col]
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    l_axes[row] = bcols[col]
  end
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    l_axes[bn + i] = bcols[emptycols[k]]
  end

  l_axis = mortar_axis(l_axes)
  L, Q = similar_output(lq_full!, A, l_axis, alg)

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    L[brow, brow], Q[brow, bcol] = MatrixAlgebraKit.initialize_output(
      lq_full!, @view!(A[bI]), alg.alg
    )
  end

  # allocate output for blocks that aren't present -- do we also fill identities here?
  for (row, col) in zip(emptyrows, emptycols)
    @view!(Q[Block(row, col)])
  end
  # also handle extra rows/cols
  for (i, k) in enumerate((length(emptyrows) + 1):length(emptycols))
    @view!(Q[Block(bm + i, emptycols[k])])
  end

  return L, Q
end

function MatrixAlgebraKit.check_input(
  ::typeof(lq_compact!), A::AbstractBlockSparseMatrix, LQ
)
  L, Q = LQ
  @assert isa(L, AbstractBlockSparseMatrix) && isa(Q, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(L) == eltype(Q)
  @assert axes(A, 1) == axes(L, 1) && axes(A, 2) == axes(Q, 2)
  @assert axes(L, 2) == axes(Q, 1)

  return nothing
end

function MatrixAlgebraKit.check_input(::typeof(lq_full!), A::AbstractBlockSparseMatrix, LQ)
  L, Q = LQ
  @assert isa(L, AbstractBlockSparseMatrix) && isa(Q, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(L) == eltype(Q)
  @assert axes(A, 1) == axes(L, 1) && axes(A, 2) == axes(Q, 2)
  @assert axes(L, 2) == axes(Q, 1)

  return nothing
end

function MatrixAlgebraKit.lq_compact!(
  A::AbstractBlockSparseMatrix, LQ, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(lq_compact!, A, LQ)
  L, Q = LQ

  # do decomposition on each block
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    lq = (@view!(L[brow, brow]), @view!(Q[brow, bcol]))
    lq′ = lq_compact!(@view!(A[bI]), lq, alg.alg)
    @assert lq === lq′ "lq_compact! might not be in-place"
  end

  # fill in identities for blocks that aren't present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  emptyrows = setdiff(1:blocksize(A, 1), browIs)
  emptycols = setdiff(1:blocksize(A, 2), bcolIs)
  # needs copyto! instead because size(::LinearAlgebra.I) doesn't work
  # Q[Block(row, col)] = LinearAlgebra.I
  for (row, col) in zip(emptyrows, emptycols)
    copyto!(@view!(Q[Block(row, col)]), LinearAlgebra.I)
  end

  return LQ
end

function MatrixAlgebraKit.lq_full!(
  A::AbstractBlockSparseMatrix, LQ, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(lq_full!, A, LQ)
  L, Q = LQ

  # do decomposition on each block
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    lq = (@view!(L[brow, brow]), @view!(Q[brow, bcol]))
    lq′ = lq_full!(@view!(A[bI]), lq, alg.alg)
    @assert lq === lq′ "lq_full! might not be in-place"
  end

  # fill in identities for blocks that aren't present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  emptyrows = setdiff(1:blocksize(A, 1), browIs)
  emptycols = setdiff(1:blocksize(A, 2), bcolIs)
  # needs copyto! instead because size(::LinearAlgebra.I) doesn't work
  # Q[Block(row, col)] = LinearAlgebra.I
  for (row, col) in zip(emptyrows, emptycols)
    copyto!(@view!(Q[Block(row, col)]), LinearAlgebra.I)
  end

  # also handle extra rows/cols
  bm = blocksize(A, 1)
  for (i, k) in enumerate((length(emptyrows) + 1):length(emptycols))
    copyto!(@view!(Q[Block(bm + i, emptycols[k])]), LinearAlgebra.I)
  end

  return LQ
end
