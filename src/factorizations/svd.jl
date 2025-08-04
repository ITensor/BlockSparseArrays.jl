using DiagonalArrays: diagonaltype
using MatrixAlgebraKit:
  MatrixAlgebraKit, check_input, default_svd_algorithm, svd_compact!, svd_full!
using TypeParameterAccessors: realtype

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
  ::typeof(svd_compact!), ::AbstractBlockSparseMatrix, ::BlockPermutedDiagonalAlgorithm
)
  return nothing
end
function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_compact!), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
)
  brows = eachblockaxis(axes(A, 1))
  bcols = eachblockaxis(axes(A, 2))
  # using the property that zip stops as soon as one of the iterators is exhausted
  s_axes = map(splat(infimum), zip(brows, bcols))
  s_axis = mortar_axis(s_axes)
  S_axes = (s_axis, s_axis)
  U, S, Vᴴ = similar_output(svd_compact!, A, S_axes, alg)

  for bI in eachblockstoredindex(A)
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    I = first(Tuple(bI)) # == last(Tuple(bI))
    U[I, I], S[I, I], Vᴴ[I, I] = MatrixAlgebraKit.initialize_output(
      svd_compact!, block, block_alg
    )
  end

  return U, S, Vᴴ
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
  ::typeof(svd_full!), ::AbstractBlockSparseMatrix, ::BlockPermutedDiagonalAlgorithm
)
  return nothing
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_full!), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
)
  U, S, Vᴴ = similar_output(svd_full!, A, axes(A), alg)

  for bI in eachblockstoredindex(A)
    block = @view!(A[bI])
    block_alg = block_algorithm(alg, block)
    I = first(Tuple(bI)) # == last(Tuple(bI))
    U[I, I], S[I, I], Vᴴ[I, I] = MatrixAlgebraKit.initialize_output(
      svd_full!, block, block_alg
    )
  end

  return U, S, Vᴴ
end

function MatrixAlgebraKit.check_input(
  ::typeof(svd_compact!),
  A::AbstractBlockSparseMatrix,
  USVᴴ,
  ::BlockPermutedDiagonalAlgorithm,
)
  @assert isblockpermuteddiagonal(A)
end

function MatrixAlgebraKit.check_input(
  ::typeof(svd_compact!), A::AbstractBlockSparseMatrix, (U, S, Vᴴ), ::BlockDiagonalAlgorithm
)
  @assert isa(U, AbstractBlockSparseMatrix) &&
    isa(S, AbstractBlockSparseMatrix) &&
    isa(Vᴴ, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(U) == eltype(Vᴴ)
  @assert real(eltype(A)) == eltype(S)
  @assert axes(A, 1) == axes(U, 1) && axes(A, 2) == axes(Vᴴ, 2)
  @assert axes(S, 1) == axes(S, 2)
  @assert isblockdiagonal(A)
  return nothing
end

function MatrixAlgebraKit.check_input(
  ::typeof(svd_full!), A::AbstractBlockSparseMatrix, USVᴴ, ::BlockPermutedDiagonalAlgorithm
)
  @assert isblockpermuteddiagonal(A)
  return nothing
end

function MatrixAlgebraKit.check_input(
  ::typeof(svd_full!), A::AbstractBlockSparseMatrix, (U, S, Vᴴ), ::BlockDiagonalAlgorithm
)
  @assert isa(U, AbstractBlockSparseMatrix) &&
    isa(S, AbstractBlockSparseMatrix) &&
    isa(Vᴴ, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(U) == eltype(Vᴴ)
  @assert real(eltype(A)) == eltype(S)
  @assert axes(A, 1) == axes(U, 1) && axes(A, 2) == axes(Vᴴ, 1) == axes(Vᴴ, 2)
  @assert axes(S, 2) == axes(A, 2)
  @assert isblockdiagonal(A)
  return nothing
end

function MatrixAlgebraKit.svd_compact!(
  A::AbstractBlockSparseMatrix, USVᴴ, alg::BlockPermutedDiagonalAlgorithm
)
  check_input(svd_compact!, A, USVᴴ, alg)

  Ad, rowperm, colperm = blockdiagonalize(A)
  Ud, S, Vᴴd = svd_compact!(Ad, BlockDiagonalAlgorithm(alg))

  inv_rowperm = Block.(invperm(Int.(rowperm)))
  U = Ud[inv_rowperm, :]

  inv_colperm = Block.(invperm(Int.(colperm)))
  Vᴴ = Vᴴd[:, inv_colperm]

  return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_compact!(
  A::AbstractBlockSparseMatrix, (U, S, Vᴴ), alg::BlockDiagonalAlgorithm
)
  check_input(svd_compact!, A, (U, S, Vᴴ), alg)

  for I in 1:min(blocksize(A)...)
    bI = Block(I, I)
    if isstored(blocks(A), CartesianIndex(I, I)) # TODO: isblockstored
      usvᴴ = (@view!(U[bI]), @view!(S[bI]), @view!(Vᴴ[bI]))
      block = @view!(A[bI])
      block_alg = block_algorithm(alg, block)
      usvᴴ′ = svd_compact!(block, usvᴴ, block_alg)
      @assert usvᴴ === usvᴴ′ "svd_compact! might not be in-place"
    else
      copyto!(@view!(U[bI]), LinearAlgebra.I)
      copyto!(@view!(Vᴴ[bI]), LinearAlgebra.I)
    end
  end

  return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_full!(
  A::AbstractBlockSparseMatrix, USVᴴ, alg::BlockPermutedDiagonalAlgorithm
)
  check_input(svd_full!, A, USVᴴ, alg)

  Ad, rowperm, colperm = blockdiagonalize(A)
  Ud, S, Vᴴd = svd_full!(Ad, BlockDiagonalAlgorithm(alg))

  inv_rowperm = Block.(invperm(Int.(rowperm)))
  U = Ud[inv_rowperm, :]

  inv_colperm = Block.(invperm(Int.(colperm)))
  Vᴴ = Vᴴd[:, inv_colperm]

  return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_full!(
  A::AbstractBlockSparseMatrix, (U, S, Vᴴ), alg::BlockDiagonalAlgorithm
)
  check_input(svd_full!, A, (U, S, Vᴴ), alg)

  for I in 1:min(blocksize(A)...)
    bI = Block(I, I)
    if isstored(blocks(A), CartesianIndex(I, I)) # TODO: isblockstored
      usvᴴ = (@view!(U[bI]), @view!(S[bI]), @view!(Vᴴ[bI]))
      block = @view!(A[bI])
      block_alg = block_algorithm(alg, block)
      usvᴴ′ = svd_full!(block, usvᴴ, block_alg)
      @assert usvᴴ === usvᴴ′ "svd_compact! might not be in-place"
    else
      copyto!(@view!(U[bI]), LinearAlgebra.I)
      copyto!(@view!(Vᴴ[bI]), LinearAlgebra.I)
    end
  end

  # Complete the unitaries for rectangular inputs
  for I in (blocksize(A, 2) + 1):blocksize(A, 1)
    copyto!(@view!(U[Block(I, I)]), LinearAlgebra.I)
  end
  for I in (blocksize(A, 1) + 1):blocksize(A, 2)
    copyto!(@view!(Vᴴ[Block(I, I)]), LinearAlgebra.I)
  end

  return U, S, Vᴴ
end
