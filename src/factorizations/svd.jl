using MatrixAlgebraKit

"""
    BlockPermutedDiagonalAlgorithm(A::MatrixAlgebraKit.AbstractAlgorithm)
  
A wrapper for `MatrixAlgebraKit.AbstractAlgorithm` that implements the wrapped algorithm on
a block-by-block basis, which is possible if the input matrix is a block-diagonal matrix.
"""
struct BlockPermutedDiagonalAlgorithm{A<:MatrixAlgebraKit.AbstractAlgorithm} <:
       MatrixAlgebraKit.AbstractAlgorithm
  alg::A
end

function MatrixAlgebraKit.default_svd_algorithm(A::AbstractBlockSparseMatrix; kwargs...)
  @assert blocktype(A) <: StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat} "unsupported type:
    $(blocktype(A))"
  alg = MatrixAlgebraKit.LAPACK_DivideAndConquer(; kwargs...)
  return BlockPermutedDiagonalAlgorithm(alg)
end

#=
Note: here I'm being generic about the matrix type, which effectively means that I'm making
some assumptions about the output type of the algorithm, ie that this will return 
Matrix{T},Diagonal{real(T)},Matrix{T}. In principle this is not guaranteed, although it
should cover most cases. The alternative is to be more specific about the matrix type and
replace the `similar` calls with explicit `BlockSparseArray` constructor calls. In that case
I can simply call `initialize_output` on the input blocks, and take whatever is returned.
We should probably discuss which way to go.
=#

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_compact!),
  A::AbstractBlockSparseMatrix,
  alg::BlockPermutedDiagonalAlgorithm,
)
  bm, bn = blocksize(A)
  bmn = min(bm, bn)

  brows = blocklengths(axes(A, 1))
  bcols = blocklengths(axes(A, 2))
  slengths = Vector{Int}(undef, bmn)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    nrows = brows[row]
    ncols = bcols[col]
    slengths[col] = min(nrows, ncols)
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    slengths[col] = min(brows[row], bcols[col])
  end

  s_axis = blockedrange(slengths)
  U = similar(A, axes(A, 1), s_axis)
  Tr = real(eltype(A))
  S = BlockSparseArray{Tr,2,Diagonal{Tr,Vector{Tr}}}(undef, (s_axis, s_axis))
  Vt = similar(A, s_axis, axes(A, 2))

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

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_full!),
  A::AbstractBlockSparseMatrix,
  alg::BlockPermutedDiagonalAlgorithm,
)
  bm, bn = blocksize(A)

  brows = blocklengths(axes(A, 1))
  slengths = copy(brows)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    nrows = brows[row]
    slengths[col] = nrows
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    slengths[col] = brows[row]
  end
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    slengths[bn + i] = brows[emptyrows[k]]
  end

  s_axis = blockedrange(slengths)
  U = similar(A, axes(A, 1), s_axis)
  Tr = real(eltype(A))
  S = similar(A, Tr, (s_axis, axes(A, 2)))
  Vt = similar(A, axes(A, 2), axes(A, 2))

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

  # TODO: implement checks on axes of S, or find better way to do this without recomputing
  # pairing
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

  # TODO: implement checks on axes of S, or find better way to do this without recomputing
  # pairing
  return nothing
end

function MatrixAlgebraKit.svd_compact!(
  A::AbstractBlockSparseMatrix,
  USVᴴ,
  alg::BlockPermutedDiagonalAlgorithm,
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
  for (row, col) in zip(emptyrows, emptycols)
    copyto!(@view!(U[Block(row, col)]), LinearAlgebra.I)
    copyto!(@view!(Vt[Block(col, col)]), LinearAlgebra.I)
  end

  return USVᴴ
end

function MatrixAlgebraKit.svd_full!(
  A::AbstractBlockSparseMatrix,
  USVᴴ,
  alg::BlockPermutedDiagonalAlgorithm,
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
