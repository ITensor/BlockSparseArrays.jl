using DiagonalArrays: diagonaltype
using MatrixAlgebraKit:
  MatrixAlgebraKit, check_input, default_svd_algorithm, svd_compact!, svd_full!, svd_vals!
using TypeParameterAccessors: realtype

function MatrixAlgebraKit.default_svd_algorithm(
  ::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  return BlockPermutedDiagonalAlgorithm() do block
    return default_svd_algorithm(block; kwargs...)
  end
end

function output_type(
  f::Union{typeof(svd_compact!),typeof(svd_full!)}, A::Type{<:AbstractMatrix{T}}
) where {T}
  USVᴴ = Base.promote_op(f, A)
  return if isconcretetype(USVᴴ)
    USVᴴ
  else
    Tuple{AbstractMatrix{T},AbstractMatrix{realtype(T)},AbstractMatrix{T}}
  end
end
function output_type(::typeof(svd_vals!), A::Type{<:AbstractMatrix{T}}) where {T}
  S = Base.promote_op(svd_vals!, A)
  return isconcretetype(S) ? S : AbstractVector{real(T)}
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

  BU, BS, BVᴴ = fieldtypes(output_type(svd_compact!, blocktype(A)))
  U = similar(A, BlockType(BU), (axes(A, 1), S_axes[1]))
  S = similar(A, BlockType(BS), S_axes)
  Vᴴ = similar(A, BlockType(BVᴴ), (S_axes[2], axes(A, 2)))

  return U, S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_full!), ::AbstractBlockSparseMatrix, ::BlockPermutedDiagonalAlgorithm
)
  return nothing
end
function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_full!), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
)
  BU, BS, BVᴴ = fieldtypes(output_type(svd_full!, blocktype(A)))
  U = similar(A, BlockType(BU), (axes(A, 1), axes(A, 1)))
  S = similar(A, BlockType(BS), axes(A))
  Vᴴ = similar(A, BlockType(BVᴴ), (axes(A, 2), axes(A, 2)))

  return U, S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_vals!), ::AbstractBlockSparseMatrix, ::BlockPermutedDiagonalAlgorithm
)
  return nothing
end
function MatrixAlgebraKit.initialize_output(
  ::typeof(svd_vals!), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
)
  brows = eachblockaxis(axes(A, 1))
  bcols = eachblockaxis(axes(A, 2))
  # using the property that zip stops as soon as one of the iterators is exhausted
  s_axes = map(splat(infimum), zip(brows, bcols))
  s_axis = mortar_axis(s_axes)

  BS = output_type(svd_vals!, blocktype(A))
  return similar(A, BlockType(BS), S_axes)
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

function MatrixAlgebraKit.check_input(
  ::typeof(svd_vals!), A::AbstractBlockSparseMatrix, S, ::BlockPermutedDiagonalAlgorithm
)
  @assert isblockpermuteddiagonal(A)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(svd_vals!), A::AbstractBlockSparseMatrix, S, ::BlockDiagonalAlgorithm
)
  @assert isa(S, AbstractBlockSparseVector)
  @assert real(eltype(A)) == eltype(S)
  @assert isblockdiagonal(A)
  return nothing
end

function MatrixAlgebraKit.svd_compact!(
  A::AbstractBlockSparseMatrix, USVᴴ, alg::BlockPermutedDiagonalAlgorithm
)
  check_input(svd_compact!, A, USVᴴ, alg)

  Ad, (invrowperm, invcolperm) = blockdiagonalize(A)
  Ud, S, Vᴴd = svd_compact!(Ad, BlockDiagonalAlgorithm(alg))
  U = transform_rows(Ud, invrowperm)
  Vᴴ = transform_cols(Vᴴd, invcolperm)

  return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_compact!(
  A::AbstractBlockSparseMatrix, (U, S, Vᴴ), alg::BlockDiagonalAlgorithm
)
  check_input(svd_compact!, A, (U, S, Vᴴ), alg)

  for I in 1:min(blocksize(A)...)
    bI = Block(I, I)
    if isstored(A, bI)
      block = @view!(A[bI])
      block_alg = block_algorithm(alg, block)
      bU, bS, bVᴴ = svd_compact!(block, block_alg)
      U[bI] = bU
      S[bI] = bS
      Vᴴ[bI] = bVᴴ
    else
      # TODO: this should be `U[bI] = LinearAlgebra.I` and `Vᴴ[bI] = LinearAlgebra.I`
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

  Ad, (invrowperm, invcolperm) = blockdiagonalize(A)
  Ud, S, Vᴴd = svd_full!(Ad, BlockDiagonalAlgorithm(alg))
  U = transform_rows(Ud, invrowperm)
  Vᴴ = transform_cols(Vᴴd, invcolperm)

  return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_full!(
  A::AbstractBlockSparseMatrix, (U, S, Vᴴ), alg::BlockDiagonalAlgorithm
)
  check_input(svd_full!, A, (U, S, Vᴴ), alg)

  for I in 1:min(blocksize(A)...)
    bI = Block(I, I)
    if isstored(A, bI)
      block = @view!(A[bI])
      block_alg = block_algorithm(alg, block)
      bU, bS, bVᴴ = svd_full!(block, block_alg)
      U[bI] = bU
      S[bI] = bS
      Vᴴ[bI] = bVᴴ
    else
      # TODO: this should be `U[bI] = LinearAlgebra.I` and `Vᴴ[bI] = LinearAlgebra.I`
      copyto!(@view!(U[bI]), LinearAlgebra.I)
      copyto!(@view!(Vᴴ[bI]), LinearAlgebra.I)
    end
  end

  # Complete the unitaries for rectangular inputs
  # TODO: this should be `U[Block(I, I)] = LinearAlgebra.I`
  for I in (blocksize(A, 2) + 1):blocksize(A, 1)
    copyto!(@view!(U[Block(I, I)]), LinearAlgebra.I)
  end
  # TODO: this should be `Vᴴ[Block(I, I)] = LinearAlgebra.I`
  for I in (blocksize(A, 1) + 1):blocksize(A, 2)
    copyto!(@view!(Vᴴ[Block(I, I)]), LinearAlgebra.I)
  end

  return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_vals!(
  A::AbstractBlockSparseMatrix, S, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(svd_vals!, A, S, alg)
  Ad, _ = blockdiagonalize(A)
  return svd_vals!(Ad, BlockDiagonalAlgorithm(alg))
end
function MatrixAlgebraKit.svd_vals!(
  A::AbstractBlockSparseMatrix, S, alg::BlockDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(svd_vals!, A, S, alg)
  for I in eachblockstoredindex(A)
    block = @view!(A[I])
    S[Tuple(I)[1]] = svd_vals!(block, block_algorithm(alg, block))
  end
  return S
end
