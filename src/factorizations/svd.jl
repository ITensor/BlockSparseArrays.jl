using MatrixAlgebraKit: MatrixAlgebraKit, svd_compact!, svd_full!

"""
    BlockPermutedDiagonalAlgorithm(A::MatrixAlgebraKit.AbstractAlgorithm)
  
A wrapper for `MatrixAlgebraKit.AbstractAlgorithm` that implements the wrapped algorithm on
a block-by-block basis, which is possible if the input matrix is a block-diagonal matrix or
a block permuted block-diagonal matrix.
"""
struct BlockPermutedDiagonalAlgorithm{A<:MatrixAlgebraKit.AbstractAlgorithm} <:
       MatrixAlgebraKit.AbstractAlgorithm
  alg::A
end

# TODO: this is a hardcoded for now to get around this function not being defined in the
# type domain
function MatrixAlgebraKit.default_svd_algorithm(A::AbstractBlockSparseMatrix; kwargs...)
  blocktype(A) <: StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat} ||
    error("unsupported type: $(blocktype(A))")
  alg = MatrixAlgebraKit.LAPACK_DivideAndConquer(; kwargs...)
  return BlockPermutedDiagonalAlgorithm(alg)
end

function similar_output(
  ::typeof(svd_compact!),
  A,
  s_axis::AbstractUnitRange,
  alg::MatrixAlgebraKit.AbstractAlgorithm,
)
  U = similar(A, axes(A, 1), s_axis)
  T = real(eltype(A))
  # TODO: this should be replaced with a more general similar function that can handle setting
  # the blocktype and element type - something like S = similar(A, BlockType(...))
  S = BlockSparseMatrix{T,Diagonal{T,Vector{T}}}(undef, (s_axis, s_axis))
  Vt = similar(A, s_axis, axes(A, 2))
  return U, S, Vt
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_compact!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)
  bmn = min(bm, bn)

  brows = eachblockaxis(axes(A, 1))
  bcols = eachblockaxis(axes(A, 2))
  s_axes = similar(brows, bmn)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    s_axes[col] = argmin(length, (brows[row], bcols[col]))
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    s_axes[col] = argmin(length, (brows[row], bcols[col]))
  end

  s_axis = mortar_axis(s_axes)
  U, S, Vt = similar_output(svd_compact!, A, s_axis, alg)

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    U[brow, bcol], S[bcol, bcol], Vt[bcol, bcol] = MatrixAlgebraKit.initialize_output(
      svd_compact!, @view!(A[bI]), alg.alg
    )
  end

  # allocate output for blocks that aren't present -- do we also fill identities here?
  for (row, col) in zip(emptyrows, emptycols)
    @view!(U[Block(row, col)])
    @view!(Vt[Block(col, col)])
  end

  return U, S, Vt
end

function similar_output(
  ::typeof(svd_full!), A, s_axis::AbstractUnitRange, alg::MatrixAlgebraKit.AbstractAlgorithm
)
  U = similar(A, axes(A, 1), s_axis)
  T = real(eltype(A))
  S = similar(A, T, (s_axis, axes(A, 2)))
  Vt = similar(A, axes(A, 2), axes(A, 2))
  return U, S, Vt
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_full!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)

  brows = eachblockaxis(axes(A, 1))
  s_axes = similar(brows)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    s_axes[col] = brows[row]
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    s_axes[col] = brows[row]
  end
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    s_axes[bn + i] = brows[emptyrows[k]]
  end

  s_axis = mortar_axis(s_axes)
  U, S, Vt = similar_output(svd_full!, A, s_axis, alg)

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    U[brow, bcol], S[bcol, bcol], Vt[bcol, bcol] = MatrixAlgebraKit.initialize_output(
      svd_full!, @view!(A[bI]), alg.alg
    )
  end

  # allocate output for blocks that aren't present -- do we also fill identities here?
  for (row, col) in zip(emptyrows, emptycols)
    @view!(U[Block(row, col)])
    @view!(Vt[Block(col, col)])
  end
  # also handle extra rows/cols
  for i in (length(emptyrows) + 1):length(emptycols)
    @view!(Vt[Block(emptycols[i], emptycols[i])])
  end
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    @view!(U[Block(emptyrows[k], bn + i)])
  end

  return U, S, Vt
end

function MatrixAlgebraKit.check_input(
  ::typeof(svd_compact!), A::AbstractBlockSparseMatrix, USVᴴ
)
  U, S, Vt = USVᴴ
  @assert isa(U, AbstractBlockSparseMatrix) &&
    isa(S, AbstractBlockSparseMatrix) &&
    isa(Vt, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(U) == eltype(Vt)
  @assert real(eltype(A)) == eltype(S)
  @assert axes(A, 1) == axes(U, 1) && axes(A, 2) == axes(Vt, 2)
  @assert axes(S, 1) == axes(S, 2)

  return nothing
end

function MatrixAlgebraKit.check_input(
  ::typeof(svd_full!), A::AbstractBlockSparseMatrix, USVᴴ
)
  U, S, Vt = USVᴴ
  @assert isa(U, AbstractBlockSparseMatrix) &&
    isa(S, AbstractBlockSparseMatrix) &&
    isa(Vt, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(U) == eltype(Vt)
  @assert real(eltype(A)) == eltype(S)
  @assert axes(A, 1) == axes(U, 1) && axes(A, 2) == axes(Vt, 1) == axes(Vt, 2)
  @assert axes(S, 2) == axes(A, 2)

  return nothing
end

function MatrixAlgebraKit.svd_compact!(
  A::AbstractBlockSparseMatrix, USVᴴ, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(svd_compact!, A, USVᴴ)
  U, S, Vt = USVᴴ

  # do decomposition on each block
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    usvᴴ = (@view!(U[brow, bcol]), @view!(S[bcol, bcol]), @view!(Vt[bcol, bcol]))
    usvᴴ′ = svd_compact!(@view!(A[bI]), usvᴴ, alg.alg)
    @assert usvᴴ === usvᴴ′ "svd_compact! might not be in-place"
  end

  # fill in identities for blocks that aren't present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  emptyrows = setdiff(1:blocksize(A, 1), browIs)
  emptycols = setdiff(1:blocksize(A, 2), bcolIs)
  # needs copyto! instead because size(::LinearAlgebra.I) doesn't work
  # U[Block(row, col)] = LinearAlgebra.I
  # Vt[Block(col, col)] = LinearAlgebra.I
  for (row, col) in zip(emptyrows, emptycols)
    copyto!(@view!(U[Block(row, col)]), LinearAlgebra.I)
    copyto!(@view!(Vt[Block(col, col)]), LinearAlgebra.I)
  end

  return USVᴴ
end

function MatrixAlgebraKit.svd_full!(
  A::AbstractBlockSparseMatrix, USVᴴ, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(svd_full!, A, USVᴴ)
  U, S, Vt = USVᴴ

  # do decomposition on each block
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    usvᴴ = (@view!(U[brow, bcol]), @view!(S[bcol, bcol]), @view!(Vt[bcol, bcol]))
    usvᴴ′ = svd_full!(@view!(A[bI]), usvᴴ, alg.alg)
    @assert usvᴴ === usvᴴ′ "svd_full! might not be in-place"
  end

  # fill in identities for blocks that aren't present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  emptyrows = setdiff(1:blocksize(A, 1), browIs)
  emptycols = setdiff(1:blocksize(A, 2), bcolIs)
  # needs copyto! instead because size(::LinearAlgebra.I) doesn't work
  # U[Block(row, col)] = LinearAlgebra.I
  # Vt[Block(col, col)] = LinearAlgebra.I
  for (row, col) in zip(emptyrows, emptycols)
    copyto!(@view!(U[Block(row, col)]), LinearAlgebra.I)
    copyto!(@view!(Vt[Block(col, col)]), LinearAlgebra.I)
  end

  # also handle extra rows/cols
  for i in (length(emptyrows) + 1):length(emptycols)
    copyto!(@view!(Vt[Block(emptycols[i], emptycols[i])]), LinearAlgebra.I)
  end
  bn = blocksize(A, 2)
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    copyto!(@view!(U[Block(emptyrows[k], bn + i)]), LinearAlgebra.I)
  end

  return USVᴴ
end
