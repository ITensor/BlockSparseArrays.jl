using MatrixAlgebraKit: MatrixAlgebraKit, qr_compact!, qr_full!

# TODO: this is a hardcoded for now to get around this function not being defined in the
# type domain
function MatrixAlgebraKit.default_qr_algorithm(A::AbstractBlockSparseMatrix; kwargs...)
  blocktype(A) <: StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat} ||
    error("unsupported type: $(blocktype(A))")
  alg = MatrixAlgebraKit.LAPACK_HouseholderQR(; kwargs...)
  return BlockPermutedDiagonalAlgorithm(alg)
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(qr_compact!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)
  bmn = min(bm, bn)

  brows = blocklengths(axes(A, 1))
  bcols = blocklengths(axes(A, 2))
  rlengths = Vector{Int}(undef, bmn)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    nrows = brows[row]
    ncols = bcols[col]
    rlengths[col] = min(nrows, ncols)
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    rlengths[col] = min(brows[row], bcols[col])
  end

  r_axis = blockedrange(rlengths)
  Q = similar(A, axes(A, 1), r_axis)
  R = similar(A, r_axis, axes(A, 2))

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    Q[brow, bcol], R[bcol, bcol] = MatrixAlgebraKit.initialize_output(
      qr_compact!, @view!(A[bI]), alg.alg
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

  brows = blocklengths(axes(A, 1))
  rlengths = copy(brows)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    nrows = brows[row]
    rlengths[col] = nrows
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    rlengths[col] = brows[row]
  end
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    rlengths[bn + i] = brows[emptyrows[k]]
  end

  r_axis = blockedrange(rlengths)
  Q = similar(A, axes(A, 1), r_axis)
  R = similar(A, r_axis, axes(A, 2))

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    Q[brow, bcol], R[bcol, bcol] = MatrixAlgebraKit.initialize_output(
      qr_full!, @view!(A[bI]), alg.alg
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
  @assert isa(Q, AbstractBlockSparseMatrix) &&
    isa(R, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(Q) == eltype(R)
  @assert axes(A, 1) == axes(Q, 1) && axes(A, 2) == axes(R, 2)
  @assert axes(Q, 2) == axes(R, 1)

  return nothing
end

function MatrixAlgebraKit.check_input(
  ::typeof(qr_full!), A::AbstractBlockSparseMatrix, QR
)
  Q, R = QR
  @assert isa(Q, AbstractBlockSparseMatrix) &&
    isa(R, AbstractBlockSparseMatrix)
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
    qr′ = qr_compact!(@view!(A[bI]), qr, alg.alg)
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
    qr′ = qr_full!(@view!(A[bI]), qr, alg.alg)
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
