using MatrixAlgebraKit: MatrixAlgebraKit, default_lq_algorithm, lq_compact!, lq_full!

function MatrixAlgebraKit.default_lq_algorithm(
  A::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  return BlockPermutedDiagonalAlgorithm() do block
    return default_lq_algorithm(block; kwargs...)
  end
end

function output_type(
  f::Union{typeof(lq_compact!),typeof(lq_full!)}, A::Type{<:AbstractMatrix{T}}
) where {T}
  LQ = Base.promote_op(f, A)
  return isconcretetype(LQ) ? LQ : Tuple{AbstractMatrix{T},AbstractMatrix{T}}
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(lq_compact!), ::AbstractBlockSparseMatrix, ::BlockPermutedDiagonalAlgorithm
)
  return nothing
end
function MatrixAlgebraKit.initialize_output(
  ::typeof(lq_compact!), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
)
  brows = eachblockaxis(axes(A, 1))
  bcols = eachblockaxis(axes(A, 2))
  # using the property that zip stops as soon as one of the iterators is exhausted
  l_axes = map(splat(infimum), zip(brows, bcols))
  l_axis = mortar_axis(l_axes)

  BL, BQ = fieldtypes(output_type(lq_compact!, blocktype(A)))
  L = similar(A, BlockType(BL), (axes(A, 1), l_axis))
  Q = similar(A, BlockType(BQ), (l_axis, axes(A, 2)))

  return L, Q
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(lq_full!), ::AbstractBlockSparseMatrix, ::BlockPermutedDiagonalAlgorithm
)
  return nothing
end
function MatrixAlgebraKit.initialize_output(
  ::typeof(lq_full!), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
)
  BL, BQ = fieldtypes(output_type(lq_full!, blocktype(A)))
  L = similar(A, BlockType(BL), (axes(A, 1), axes(A, 2)))
  Q = similar(A, BlockType(BQ), (axes(A, 2), axes(A, 2)))
  return L, Q
end

function MatrixAlgebraKit.check_input(
  ::typeof(lq_compact!), A::AbstractBlockSparseMatrix, LQ, ::BlockPermutedDiagonalAlgorithm
)
  @assert isblockpermuteddiagonal(A)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(lq_compact!), A::AbstractBlockSparseMatrix, (L, Q), ::BlockDiagonalAlgorithm
)
  @assert isa(L, AbstractBlockSparseMatrix) && isa(Q, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(L) == eltype(Q)
  @assert axes(A, 1) == axes(L, 1) && axes(A, 2) == axes(Q, 2)
  @assert axes(L, 2) == axes(Q, 1)
  @assert isblockdiagonal(A)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(lq_full!), A::AbstractBlockSparseMatrix, LQ, ::BlockPermutedDiagonalAlgorithm
)
  @assert isblockpermuteddiagonal(A)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(lq_full!), A::AbstractBlockSparseMatrix, (L, Q), ::BlockDiagonalAlgorithm
)
  @assert isa(L, AbstractBlockSparseMatrix) && isa(Q, AbstractBlockSparseMatrix)
  @assert eltype(A) == eltype(L) == eltype(Q)
  @assert axes(A, 1) == axes(L, 1) && axes(A, 2) == axes(Q, 2)
  @assert axes(L, 2) == axes(Q, 1)
  @assert isblockdiagonal(A)
  return nothing
end

function MatrixAlgebraKit.lq_compact!(
  A::AbstractBlockSparseMatrix, LQ, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(lq_compact!, A, LQ, alg)
  Ad, (invrowperm, invcolperm) = blockdiagonalize(A)
  Ld, Qd = lq_compact!(Ad, BlockDiagonalAlgorithm(alg))
  L = transform_rows(Ld, invrowperm)
  Q = transform_cols(Qd, invcolperm)
  return L, Q
end
function MatrixAlgebraKit.lq_compact!(
  A::AbstractBlockSparseMatrix, (L, Q), alg::BlockDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(lq_compact!, A, (L, Q), alg)

  # do decomposition on each block
  for bI in blockdiagindices(A)
    if isstored(A, bI)
      block = @view!(A[bI])
      block_alg = block_algorithm(alg, block)
      bL, bQ = lq_compact!(block, block_alg)
      L[bI] = bL
      Q[bI] = bQ
    else
      # TODO: this should be `Q[bI] = LinearAlgebra.I`
      copyto!(@view!(Q[bI]), LinearAlgebra.I)
    end
  end

  return L, Q
end

function MatrixAlgebraKit.lq_full!(
  A::AbstractBlockSparseMatrix, LQ, alg::BlockPermutedDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(lq_full!, A, LQ, alg)
  Ad, (invrowperm, invcolperm) = blockdiagonalize(A)
  Ld, Qd = lq_full!(Ad, BlockDiagonalAlgorithm(alg))
  L = transform_rows(Ld, invrowperm)
  Q = transform_cols(Qd, invcolperm)
  return L, Q
end
function MatrixAlgebraKit.lq_full!(
  A::AbstractBlockSparseMatrix, (L, Q), alg::BlockDiagonalAlgorithm
)
  MatrixAlgebraKit.check_input(lq_full!, A, (L, Q), alg)

  for bI in blockdiagindices(A)
    if isstored(A, bI)
      block = @view!(A[bI])
      block_alg = block_algorithm(alg, block)
      bL, bQ = lq_full!(block, block_alg)
      L[bI] = bL
      Q[bI] = bQ
    else
      # TODO: this should be `Q[bI] = LinearAlgebra.I`
      copyto!(@view!(Q[bI]), LinearAlgebra.I)
    end
  end

  return L, Q
end
