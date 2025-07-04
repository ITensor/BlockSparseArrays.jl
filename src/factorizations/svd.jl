using DiagonalArrays: diagonaltype
using MatrixAlgebraKit:
  MatrixAlgebraKit, check_input, default_svd_algorithm, svd_compact!, svd_full!
using TypeParameterAccessors: realtype

"""
    BlockPermutedDiagonalAlgorithm(A::MatrixAlgebraKit.AbstractAlgorithm)
  
A wrapper for `MatrixAlgebraKit.AbstractAlgorithm` that implements the wrapped algorithm on
a block-by-block basis, which is possible if the input matrix is a block-diagonal matrix or
a block permuted block-diagonal matrix.
"""
struct BlockPermutedDiagonalAlgorithm{F} <: MatrixAlgebraKit.AbstractAlgorithm
  falg::F
end
function block_algorithm(alg::BlockPermutedDiagonalAlgorithm, a::AbstractMatrix)
  return block_algorithm(alg, typeof(a))
end
function block_algorithm(alg::BlockPermutedDiagonalAlgorithm, A::Type{<:AbstractMatrix})
  return alg.falg(A)
end

function MatrixAlgebraKit.default_svd_algorithm(
  ::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  return BlockPermutedDiagonalAlgorithm() do block
    return default_svd_algorithm(block; kwargs...)
  end
end

function output_type(::typeof(svd_compact!), A::Type{<:AbstractMatrix{T}}) where {T}
  USVᴴ = Base.promote_op(svd_compact!, A)
  !isconcretetype(USVᴴ) &&
    return Tuple{AbstractMatrix{T},AbstractMatrix{realtype(T)},AbstractMatrix{T}}
  return USVᴴ
end

function similar_output(
  ::typeof(svd_compact!), A, S_axes, alg::MatrixAlgebraKit.AbstractAlgorithm
)
  BU, BS, BVᴴ = fieldtypes(output_type(svd_compact!, blocktype(A)))
  U = similar(A, BlockType(BU), (axes(A, 1), S_axes[1]))
  S = similar(A, BlockType(BS), S_axes)
  Vᴴ = similar(A, BlockType(BVᴴ), (S_axes[2], axes(A, 2)))
  return U, S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_compact!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)
  bmn = min(bm, bn)

  brows = eachblockaxis(axes(A, 1))
  bcols = eachblockaxis(axes(A, 2))
  u_axes = similar(brows, bmn)
  v_axes = similar(brows, bmn)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    u_axes[col] = infimum(brows[row], bcols[col])
    v_axes[col] = infimum(bcols[col], brows[row])
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    u_axes[col] = infimum(brows[row], bcols[col])
    v_axes[col] = infimum(bcols[col], brows[row])
  end

  u_axis = mortar_axis(u_axes)
  v_axis = mortar_axis(v_axes)
  S_axes = (u_axis, v_axis)
  U, S, Vt = similar_output(svd_compact!, A, S_axes, alg)

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    U[brow, bcol], S[bcol, bcol], Vt[bcol, bcol] = MatrixAlgebraKit.initialize_output(
      svd_compact!, block, block_alg
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
  ::typeof(svd_full!), A, S_axes, alg::MatrixAlgebraKit.AbstractAlgorithm
)
  U = similar(A, axes(A, 1), S_axes[1])
  T = real(eltype(A))
  S = similar(A, T, S_axes)
  Vt = similar(A, S_axes[2], axes(A, 2))
  return U, S, Vt
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_full!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  bm, bn = blocksize(A)

  brows = eachblockaxis(axes(A, 1))
  u_axes = similar(brows)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    u_axes[col] = brows[row]
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    u_axes[col] = brows[row]
  end
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    u_axes[bn + i] = brows[emptyrows[k]]
  end

  u_axis = mortar_axis(u_axes)
  S_axes = (u_axis, axes(A, 2))
  U, S, Vt = similar_output(svd_full!, A, S_axes, alg)

  # allocate output
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    U[brow, bcol], S[bcol, bcol], Vt[bcol, bcol] = MatrixAlgebraKit.initialize_output(
      svd_full!, block, block_alg
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
  ::typeof(svd_compact!), A::AbstractBlockSparseMatrix, (U, S, Vᴴ)
)
  @assert isa(U, AbstractBlockSparseMatrix) &&
    isa(S, AbstractBlockSparseMatrix) &&
    isa(Vᴴ, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(U) == eltype(Vᴴ)
  @assert real(eltype(A)) == eltype(S)
  @assert axes(A, 1) == axes(U, 1) && axes(A, 2) == axes(Vᴴ, 2)
  @assert axes(S, 1) == axes(S, 2)
  return nothing
end

function MatrixAlgebraKit.check_input(
  ::typeof(svd_full!), A::AbstractBlockSparseMatrix, (U, S, Vᴴ)
)
  @assert isa(U, AbstractBlockSparseMatrix) &&
    isa(S, AbstractBlockSparseMatrix) &&
    isa(Vᴴ, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(U) == eltype(Vᴴ)
  @assert real(eltype(A)) == eltype(S)
  @assert axes(A, 1) == axes(U, 1) && axes(A, 2) == axes(Vᴴ, 1) == axes(Vᴴ, 2)
  @assert axes(S, 2) == axes(A, 2)
  return nothing
end

function MatrixAlgebraKit.svd_compact!(
  A::AbstractBlockSparseMatrix, (U, S, Vᴴ), alg::BlockPermutedDiagonalAlgorithm
)
  check_input(svd_compact!, A, (U, S, Vᴴ))

  # do decomposition on each block
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    usvᴴ = (@view!(U[brow, bcol]), @view!(S[bcol, bcol]), @view!(Vᴴ[bcol, bcol]))
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    usvᴴ′ = svd_compact!(block, usvᴴ, block_alg)
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
  # Vᴴ[Block(col, col)] = LinearAlgebra.I
  for (row, col) in zip(emptyrows, emptycols)
    copyto!(@view!(U[Block(row, col)]), LinearAlgebra.I)
    copyto!(@view!(Vᴴ[Block(col, col)]), LinearAlgebra.I)
  end

  return (U, S, Vᴴ)
end

function MatrixAlgebraKit.svd_full!(
  A::AbstractBlockSparseMatrix, (U, S, Vᴴ), alg::BlockPermutedDiagonalAlgorithm
)
  check_input(svd_full!, A, (U, S, Vᴴ))

  # do decomposition on each block
  for bI in eachblockstoredindex(A)
    brow, bcol = Tuple(bI)
    usvᴴ = (@view!(U[brow, bcol]), @view!(S[bcol, bcol]), @view!(Vᴴ[bcol, bcol]))
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    usvᴴ′ = svd_full!(block, usvᴴ, block_alg)
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
    copyto!(@view!(Vᴴ[Block(col, col)]), LinearAlgebra.I)
  end

  # also handle extra rows/cols
  for i in (length(emptyrows) + 1):length(emptycols)
    copyto!(@view!(Vᴴ[Block(emptycols[i], emptycols[i])]), LinearAlgebra.I)
  end
  bn = blocksize(A, 2)
  for (i, k) in enumerate((length(emptycols) + 1):length(emptyrows))
    copyto!(@view!(U[Block(emptyrows[k], bn + i)]), LinearAlgebra.I)
  end

  return (U, S, Vᴴ)
end
