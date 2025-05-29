using MatrixAlgebraKit:
  MatrixAlgebraKit,
  default_svd_algorithm,
  left_null!,
  left_null_svd!,
  left_orth!,
  left_polar!,
  lq_compact!,
  null_truncation_strategy,
  qr_compact!,
  right_null!,
  right_null_svd!,
  right_orth!,
  right_polar!,
  select_algorithm,
  svd_compact!

function MatrixAlgebraKit.initialize_output(
  ::typeof(left_orth!), A::AbstractBlockSparseMatrix
)
  return nothing
end
function MatrixAlgebraKit.check_input(::typeof(left_orth!), A::AbstractBlockSparseMatrix, F)
  !isnothing(F) && throw(
    ArgumentError(
      "`left_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  return nothing
end

function MatrixAlgebraKit.left_orth_qr!(A::AbstractBlockSparseMatrix, F, alg)
  !isnothing(F) && throw(
    ArgumentError(
      "`left_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  alg′ = select_algorithm(qr_compact!, A, alg)
  return qr_compact!(A, alg′)
end
function MatrixAlgebraKit.left_orth_polar!(A::AbstractBlockSparseMatrix, F, alg)
  !isnothing(F) && throw(
    ArgumentError(
      "`left_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  alg′ = select_algorithm(left_polar!, A, alg)
  return left_polar!(A, alg′)
end
function MatrixAlgebraKit.left_orth_svd!(
  A::AbstractBlockSparseMatrix, F, alg, trunc::Nothing=nothing
)
  !isnothing(F) && throw(
    ArgumentError(
      "`left_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  alg′ = select_algorithm(svd_compact!, A, alg)
  U, S, Vᴴ = svd_compact!(A, alg′)
  return U, S * Vᴴ
end
function MatrixAlgebraKit.left_orth_svd!(A::AbstractBlockSparseMatrix, F, alg, trunc)
  !isnothing(F) && throw(
    ArgumentError(
      "`left_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  alg′ = select_algorithm(svd_compact!, A, alg)
  alg_trunc = select_algorithm(svd_trunc!, A, alg′; trunc)
  U, S, Vᴴ = svd_trunc!(A, alg_trunc)
  return U, S * Vᴴ
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(right_orth!), A::AbstractBlockSparseMatrix
)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(right_orth!), A::AbstractBlockSparseMatrix, F::Nothing
)
  !isnothing(F) && throw(
    ArgumentError(
      "`right_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  return nothing
end

function MatrixAlgebraKit.right_orth_lq!(A::AbstractBlockSparseMatrix, F, alg)
  !isnothing(F) && throw(
    ArgumentError(
      "`right_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  alg′ = select_algorithm(lq_compact!, A, alg)
  return lq_compact!(A, alg′)
end
function MatrixAlgebraKit.right_orth_polar!(A::AbstractBlockSparseMatrix, F, alg)
  !isnothing(F) && throw(
    ArgumentError(
      "`right_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  alg′ = select_algorithm(right_polar!, A, alg)
  return right_polar!(A, alg′)
end
function MatrixAlgebraKit.right_orth_svd!(
  A::AbstractBlockSparseMatrix, F, alg, trunc::Nothing=nothing
)
  !isnothing(F) && throw(
    ArgumentError(
      "`right_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  alg′ = select_algorithm(svd_compact!, A, alg)
  U, S, Vᴴ = svd_compact!(A, alg′)
  return U * S, Vᴴ
end
function MatrixAlgebraKit.right_orth_svd!(A::AbstractBlockSparseMatrix, F, alg, trunc)
  !isnothing(F) && throw(
    ArgumentError(
      "`right_orth!` on block sparse matrices does not support specifying the output"
    ),
  )
  alg′ = select_algorithm(svd_compact!, A, alg)
  alg_trunc = select_algorithm(svd_trunc!, A, alg′; trunc)
  U, S, Vᴴ = svd_trunc!(A, alg_trunc)
  return U * S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(left_null!), A::AbstractBlockSparseMatrix
)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(left_null!), A::AbstractBlockSparseMatrix, N::Nothing
)
  return nothing
end
function MatrixAlgebraKit.left_null_qr!(A::AbstractBlockSparseMatrix, N, alg)
  return left_null_svd!(A, N, default_svd_algorithm(A))
end
function MatrixAlgebraKit.left_null_svd!(
  A::AbstractBlockSparseMatrix, N, alg, trunc::Nothing
)
  return left_null_svd!(A, N, alg, null_truncation_strategy(; atol=0, rtol=0))
end
function MatrixAlgebraKit.truncate!(
  ::typeof(left_null!),
  (U, S)::Tuple{AbstractBlockSparseMatrix,AbstractBlockSparseMatrix},
  strategy::TruncationStrategy,
)
  return error("Not implemented.")
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(right_null!), A::AbstractBlockSparseMatrix
)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(right_null!), A::AbstractBlockSparseMatrix, N::Nothing
)
  return nothing
end
function MatrixAlgebraKit.right_null_lq!(A::AbstractBlockSparseMatrix, N, alg)
  return error("Not implement.")
end
function MatrixAlgebraKit.truncate!(
  ::typeof(right_null!),
  (S, Vᴴ)::Tuple{AbstractBlockSparseMatrix,AbstractBlockSparseMatrix},
  strategy::TruncationStrategy,
)
  return error("Not implemented.")
end
