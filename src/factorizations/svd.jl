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
  s_axes = similar(brows, bmn)

  # fill in values for blocks that are present
  bIs = sort!(collect(eachblockstoredindex(A)); by=Int ∘ last ∘ Tuple)
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for (I, bI) in enumerate(bIs)
    row, col = Int.(Tuple(bI))
    s_axes[I] = infimum(brows[row], bcols[col])
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  emptycols = setdiff(1:bn, bcolIs)
  for (row, col) in zip(emptyrows, emptycols)
    s_axes[col] = infimum(brows[row], bcols[col])
  end

  s_axis = mortar_axis(s_axes)
  S_axes = (s_axis, s_axis)
  U, S, Vt = similar_output(svd_compact!, A, S_axes, alg)

  # allocate output
  for (I, bI) in enumerate(bIs)
    brow, bcol = Tuple(bI)
    bcol′ = Block(I)
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    U[brow, bcol′], S[bcol′, bcol′], Vt[bcol′, bcol] = MatrixAlgebraKit.initialize_output(
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
  bcols = eachblockaxis(axes(A, 2))
  u_axes = similar(brows, bm)
  v_axes = similar(bcols, bn)

  # fill in values for blocks that are present
  bIs = sort!(collect(eachblockstoredindex(A)), by=Int ∘ last ∘ Tuple)
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for (I, bI) in enumerate(bIs)
    row, col = Int.(Tuple(bI))
    u_axes[I] = brows[row]
    v_axes[I] = bcols[col]
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = setdiff(1:bm, browIs)
  u_axes[length(bIs) .+ (1:length(emptyrows))] .= brows[emptyrows]
  emptycols = setdiff(1:bn, bcolIs)
  v_axes[length(bIs) .+ (1:length(emptycols))] .= bcols[emptycols]
  
  u_axis = mortar_axis(u_axes)
  v_axis = mortar_axis(@show v_axes)
  S_axes = (u_axis, v_axis)
  U, S, Vt = similar_output(svd_full!, A, S_axes, alg)

  # allocate output
  for (I, bI) in enumerate(bIs)
    brow, bcol = Tuple(bI)
    bcol′ = Block(I)
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    U[brow, bcol′], S[bcol′, bcol′], Vt[bcol′, bcol] = MatrixAlgebraKit.initialize_output(
      svd_full!, block, block_alg
    )
  end

  # allocate output for blocks that aren't present -- do we also fill identities here?
  for (I, row) in enumerate(emptyrows)
    @view!(U[Block(row, I)])
  end
  for (I, col) in enumerate(emptycols)
    @view!(Vt[Block(I, col)])
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
  @assert axes(A, 1) == axes(U, 1) && axes(A, 2) == axes(Vᴴ, 2)
  return nothing
end

function MatrixAlgebraKit.svd_compact!(
  A::AbstractBlockSparseMatrix, (U, S, Vᴴ), alg::BlockPermutedDiagonalAlgorithm
)
  check_input(svd_compact!, A, (U, S, Vᴴ))

  # do decomposition on each block
  bIs = sort!(collect(eachblockstoredindex(A)); by=Int ∘ last ∘ Tuple)
  for (I, bI) in enumerate(bIs)
    brow, bcol = Tuple(bI)
    bcol′ = Block(I)
    usvᴴ = (@view!(U[brow, bcol′]), @view!(S[bcol′, bcol′]), @view!(Vᴴ[bcol′, bcol]))
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    usvᴴ′ = svd_compact!(block, usvᴴ, block_alg)
    @assert usvᴴ === usvᴴ′ "svd_compact! might not be in-place"
  end

  # fill in identities for blocks that aren't present
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
  bIs = sort!(collect(eachblockstoredindex(A)); by=Int ∘ last ∘ Tuple)
  for (I, bI) in enumerate(bIs)
    brow, bcol = Tuple(bI)
    bcol′ = Block(I)
    usvᴴ = (@view!(U[brow, bcol′]), @view!(S[bcol′, bcol′]), @view!(Vᴴ[bcol′, bcol]))
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    usvᴴ′ = svd_full!(block, usvᴴ, block_alg)
    @assert usvᴴ === usvᴴ′ "svd_full! might not be in-place"
  end

  # fill in identities for blocks that aren't present
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  emptyrows = setdiff(1:blocksize(A, 1), browIs)
  emptycols = setdiff(1:blocksize(A, 2), bcolIs)
  # needs copyto! instead because size(::LinearAlgebra.I) doesn't work
  # U[Block(row, col)] = LinearAlgebra.I
  # Vt[Block(col, col)] = LinearAlgebra.I
  for (I, row) in enumerate(emptyrows)
    copyto!(@view!(U[Block(row, length(bIs) + I)]), LinearAlgebra.I)
  end
  for (I, col) in enumerate(emptycols)
    copyto!(@view!(Vᴴ[Block(length(bIs) + I, col)]), LinearAlgebra.I)
  end

  return (U, S, Vᴴ)
end
