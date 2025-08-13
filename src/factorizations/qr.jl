using MatrixAlgebraKit: MatrixAlgebraKit, default_qr_algorithm, qr_compact!, qr_full!

function MatrixAlgebraKit.default_qr_algorithm(
  ::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  return BlockPermutedDiagonalAlgorithm() do block
    return default_qr_algorithm(block; kwargs...)
  end
end

function output_type(
  f::Union{typeof(qr_compact!),typeof(qr_full!)}, A::Type{<:AbstractMatrix{T}}
) where {T}
  QR = Base.promote_op(f, A)
  return isconcretetype(QR) ? QR : Tuple{AbstractMatrix{T},AbstractMatrix{T}}
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(qr_compact!), ::AbstractBlockSparseMatrix, ::BlockPermutedDiagonalAlgorithm
)
  return nothing
end
function MatrixAlgebraKit.initialize_output(
  ::typeof(qr_compact!), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
)
  brows = eachblockaxis(axes(A, 1))
  bcols = eachblockaxis(axes(A, 2))
  # using the property that zip stops as soon as one of the iterators is exhausted
  r_axes = map(splat(infimum), zip(brows, bcols))
  r_axis = mortar_axis(r_axes)

  BQ, BR = fieldtypes(output_type(qr_compact!, blocktype(A)))
  Q = similar(A, BlockType(BQ), (axes(A, 1), r_axis))
  R = similar(A, BlockType(BR), (r_axis, axes(A, 2)))

  return Q, R
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(qr_full!), ::AbstractBlockSparseMatrix, ::BlockPermutedDiagonalAlgorithm
)
  return nothing
end
function MatrixAlgebraKit.initialize_output(
  ::typeof(qr_full!), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
)
  BQ, BR = fieldtypes(output_type(qr_full!, blocktype(A)))
  Q = similar(A, BlockType(BQ), (axes(A, 1), axes(A, 1)))
  R = similar(A, BlockType(BR), (axes(A, 1), axes(A, 2)))
  return Q, R
end

function MatrixAlgebraKit.check_input(
  ::typeof(qr_compact!), A::AbstractBlockSparseMatrix, QR, ::BlockPermutedDiagonalAlgorithm
)
  @assert isblockpermuteddiagonal(A)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(qr_compact!), A::AbstractBlockSparseMatrix, (Q, R), ::BlockDiagonalAlgorithm
)
  @assert isa(Q, AbstractBlockSparseMatrix) && isa(R, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(Q) == eltype(R)
  @assert axes(A, 1) == axes(Q, 1) && axes(A, 2) == axes(R, 2)
  @assert axes(Q, 2) == axes(R, 1)
  @assert isblockdiagonal(A)
  return nothing
end

function MatrixAlgebraKit.check_input(
  ::typeof(qr_full!), A::AbstractBlockSparseMatrix, QR, ::BlockPermutedDiagonalAlgorithm
)
  @assert isblockpermuteddiagonal(A)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(qr_full!), A::AbstractBlockSparseMatrix, (Q, R), ::BlockDiagonalAlgorithm
)
  @assert isa(Q, AbstractBlockSparseMatrix) && isa(R, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(Q) == eltype(R)
  @assert axes(A, 1) == axes(Q, 1) && axes(A, 2) == axes(R, 2)
  @assert axes(Q, 2) == axes(R, 1)
  @assert isblockdiagonal(A)
  return nothing
end

function MatrixAlgebraKit.qr_compact!(
  A::AbstractBlockSparseMatrix, QR, alg::BlockPermutedDiagonalAlgorithm
)
  check_input(qr_compact!, A, QR, alg)
  Ad, transform_rows, transform_cols = blockdiagonalize(A)
  Qd, Rd = qr_compact!(Ad, BlockDiagonalAlgorithm(alg))
  Q = transform_rows(Qd)
  R = transform_cols(Rd)
  return Q, R
end

function MatrixAlgebraKit.qr_compact!(
  A::AbstractBlockSparseMatrix, (Q, R), alg::BlockDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(qr_compact!, A, (Q, R), alg)

  # do decomposition on each block
  for I in 1:min(blocksize(A)...)
    bI = Block(I, I)
    if isstored(blocks(A), CartesianIndex(I, I)) # TODO: isblockstored
      block = @view!(A[bI])
      block_alg = block_algorithm(alg, block)
      bQ, bR = qr_compact!(block, block_alg)
      Q[bI] = bQ
      R[bI] = bR
    else
      copyto!(@view!(Q[bI]), LinearAlgebra.I)
    end
  end

  return Q, R
end

function MatrixAlgebraKit.qr_full!(
  A::AbstractBlockSparseMatrix, QR, alg::BlockPermutedDiagonalAlgorithm
)
  check_input(qr_full!, A, QR, alg)
  Ad, transform_rows, transform_cols = blockdiagonalize(A)
  Qd, Rd = qr_full!(Ad, BlockDiagonalAlgorithm(alg))
  Q = transform_rows(Qd)
  R = transform_cols(Rd)
  return Q, R
end

function MatrixAlgebraKit.qr_full!(
  A::AbstractBlockSparseMatrix, (Q, R), alg::BlockDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(qr_full!, A, (Q, R), alg)

  for I in 1:min(blocksize(A)...)
    bI = Block(I, I)
    if isstored(blocks(A), CartesianIndex(I, I)) # TODO: isblockstored
      block = @view!(A[bI])
      block_alg = block_algorithm(alg, block)
      bQ, bR = qr_full!(block, block_alg)
      Q[bI] = bQ
      R[bI] = bR
    else
      copyto!(@view!(Q[bI]), LinearAlgebra.I)
    end
  end

  return Q, R
end
