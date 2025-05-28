using MatrixAlgebraKit:
  MatrixAlgebraKit,
  left_orth!,
  left_polar!,
  lq_compact!,
  qr_compact!,
  right_orth!,
  right_polar!,
  select_algorithm,
  svd_compact!

function MatrixAlgebraKit.initialize_output(
  ::typeof(left_orth!), A::AbstractBlockSparseMatrix
)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(left_orth!), A::AbstractBlockSparseMatrix, F::Nothing
)
  return nothing
end

function MatrixAlgebraKit.left_orth_qr!(A::AbstractBlockSparseMatrix, F, alg)
  alg′ = select_algorithm(qr_compact!, A, alg)
  return qr_compact!(A, alg′)
end
function MatrixAlgebraKit.left_orth_polar!(A::AbstractBlockSparseMatrix, F, alg)
  alg′ = select_algorithm(left_polar!, A, alg)
  return left_polar!(A, alg′)
end
function MatrixAlgebraKit.left_orth_svd!(
  A::AbstractBlockSparseMatrix, F, alg, trunc::Nothing=nothing
)
  alg′ = select_algorithm(svd_compact!, A, alg)
  U, S, Vᴴ = svd_compact!(A, alg′)
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
  return nothing
end

function MatrixAlgebraKit.right_orth_lq!(A::AbstractBlockSparseMatrix, F, alg)
  alg′ = select_algorithm(lq_compact!, A, alg)
  return lq_compact!(A, alg′)
end
function MatrixAlgebraKit.right_orth_polar!(A::AbstractBlockSparseMatrix, F, alg)
  alg′ = select_algorithm(right_polar!, A, alg)
  return right_polar!(A, alg′)
end
function MatrixAlgebraKit.right_orth_svd!(
  A::AbstractBlockSparseMatrix, F, alg, trunc::Nothing=nothing
)
  alg′ = select_algorithm(svd_compact!, A, alg)
  U, S, Vᴴ = svd_compact!(A, alg′)
  return U * S, Vᴴ
end
function MatrixAlgebraKit.right_orth_svd!(A::AbstractBlockSparseMatrix, F, alg, trunc)
  alg′ = select_algorithm(svd_compact!, A, alg)
  alg_trunc = select_algorithm(svd_trunc!, A, alg′; trunc)
  U, S, Vᴴ = svd_trunc!(A, alg_trunc)
  return U * S, Vᴴ
end
