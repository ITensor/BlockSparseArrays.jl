using MatrixAlgebraKit:
  MatrixAlgebraKit, default_qr_algorithm, lq_compact!, lq_full!, qr_compact!, qr_full!

function MatrixAlgebraKit.default_qr_algorithm(
  ::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  return BlockPermutedDiagonalAlgorithm() do block
    return default_qr_algorithm(block; kwargs...)
  end
end

function similar_output(
  ::typeof(qr_compact!), A, R_axis, alg::MatrixAlgebraKit.AbstractAlgorithm
)
  Q = similar(A, axes(A, 1), R_axis)
  R = similar(A, R_axis, axes(A, 2))
  return Q, R
end

function similar_output(
  ::typeof(qr_full!), A, R_axis, alg::MatrixAlgebraKit.AbstractAlgorithm
)
  Q = similar(A, axes(A, 1), R_axis)
  R = similar(A, R_axis, axes(A, 2))
  return Q, R
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(qr_compact!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)
  bmn = min(bm, bn)

  brows = eachblockaxis(axes(A, 1))
  bcols = eachblockaxis(axes(A, 2))
  r_axes = similar(brows, bmn)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    len = minimum(length, (brows[row], bcols[col]))
    r_axes[col] = brows[row][Base.OneTo(len)]
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    len = minimum(length, (brows[row], bcols[col]))
    r_axes[col] = brows[row][Base.OneTo(len)]
  end

  r_axis = mortar_axis(r_axes)
  Q, R = similar_output(qr_compact!, A, r_axis, alg)

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    Q[brow, bcol], R[bcol, bcol] = MatrixAlgebraKit.initialize_output(
      qr_compact!, block, block_alg
    )
  end

  # allocate output for blocks that aren't present -- do we also fill identities here?
  for (row, col) in zip(emptyrows, emptycols)
    @view!(Q[Block(row, col)])
  end

  return Q, R
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(qr_full!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)

  brows = eachblockaxis(axes(A, 1))
  r_axes = copy(brows)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    r_axes[col] = brows[row]
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    r_axes[col] = brows[row]
  end
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    r_axes[bn + i] = brows[emptyrows[k]]
  end

  r_axis = mortar_axis(r_axes)
  Q, R = similar_output(qr_full!, A, r_axis, alg)

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    Q[brow, bcol], R[bcol, bcol] = MatrixAlgebraKit.initialize_output(
      qr_full!, block, block_alg
    )
  end

  # allocate output for blocks that aren't present -- do we also fill identities here?
  for (row, col) in zip(emptyrows, emptycols)
    @view!(Q[Block(row, col)])
  end
  # also handle extra rows/cols
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    @view!(Q[Block(emptyrows[k], bn + i)])
  end

  return Q, R
end

function MatrixAlgebraKit.check_input(
  ::typeof(qr_compact!), A::AbstractBlockSparseMatrix, QR
)
  Q, R = QR
  @assert isa(Q, AbstractBlockSparseMatrix) && isa(R, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(Q) == eltype(R)
  @assert axes(A, 1) == axes(Q, 1) && axes(A, 2) == axes(R, 2)
  @assert axes(Q, 2) == axes(R, 1)

  return nothing
end

function MatrixAlgebraKit.check_input(::typeof(qr_full!), A::AbstractBlockSparseMatrix, QR)
  Q, R = QR
  @assert isa(Q, AbstractBlockSparseMatrix) && isa(R, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(Q) == eltype(R)
  @assert axes(A, 1) == axes(Q, 1) && axes(A, 2) == axes(R, 2)
  @assert axes(Q, 2) == axes(R, 1)

  return nothing
end

function MatrixAlgebraKit.qr_compact!(
  A::AbstractBlockSparseMatrix, QR, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(qr_compact!, A, QR)
  Q, R = QR

  # do decomposition on each block
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    qr = (@view!(Q[brow, bcol]), @view!(R[bcol, bcol]))
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    qr′ = qr_compact!(block, qr, block_alg)
    @assert qr === qr′ "qr_compact! might not be in-place"
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

  return QR
end

function MatrixAlgebraKit.qr_full!(
  A::AbstractBlockSparseMatrix, QR, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(qr_full!, A, QR)
  Q, R = QR

  # do decomposition on each block
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    qr = (@view!(Q[brow, bcol]), @view!(R[bcol, bcol]))
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    qr′ = qr_full!(block, qr, block_alg)
    @assert qr === qr′ "qr_full! might not be in-place"
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
  bn = blocksize(A, 2)
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    copyto!(@view!(Q[Block(emptyrows[k], bn + i)]), LinearAlgebra.I)
  end

  return QR
end
