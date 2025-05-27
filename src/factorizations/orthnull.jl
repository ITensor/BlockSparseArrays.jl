using MatrixAlgebraKit:
  MatrixAlgebraKit,
  left_orth_polar!,
  left_orth_qr!,
  left_orth_svd!,
  left_polar!,
  lq_compact!,
  qr_compact!,
  right_orth_lq!,
  right_orth_polar!,
  right_orth_svd!,
  right_polar!,
  select_algorithm,
  svd_compact!

function MatrixAlgebraKit.left_orth!(
  A::AbstractBlockSparseMatrix;
  trunc=nothing,
  kind=isnothing(trunc) ? :qr : :svd,
  alg_qr=(; positive=true),
  alg_polar=(;),
  alg_svd=(;),
)
  if !isnothing(trunc) && kind != :svd
    throw(ArgumentError("truncation not supported for `left_orth` with `kind=$kind`"))
  end
  if kind == :qr
    return left_orth_qr!(A, alg_qr)
  elseif kind == :polar
    return left_orth_polar!(A, alg_polar)
  elseif kind == :svd
    return left_orth_svd!(A, alg_svd, trunc)
  else
    throw(ArgumentError("`left_orth` received unknown value `kind = $kind`"))
  end
end
function MatrixAlgebraKit.left_orth_qr!(A::AbstractBlockSparseMatrix, alg)
  alg′ = select_algorithm(qr_compact!, A, alg)
  return qr_compact!(A, alg′)
end
function MatrixAlgebraKit.left_orth_polar!(A::AbstractBlockSparseMatrix, alg)
  alg′ = select_algorithm(left_polar!, A, alg)
  return left_polar!(A, alg′)
end
function MatrixAlgebraKit.left_orth_svd!(
  A::AbstractBlockSparseMatrix, alg, trunc::Nothing=nothing
)
  alg′ = select_algorithm(svd_compact!, A, alg)
  U, S, Vᴴ = svd_compact!(A, alg′)
  return U, S * Vᴴ
end

function MatrixAlgebraKit.right_orth!(
  A::AbstractBlockSparseMatrix;
  trunc=nothing,
  kind=isnothing(trunc) ? :lq : :svd,
  alg_lq=(; positive=true),
  alg_polar=(;),
  alg_svd=(;),
)
  if !isnothing(trunc) && kind != :svd
    throw(ArgumentError("truncation not supported for `right_orth` with `kind=$kind`"))
  end
  if kind == :qr
    # TODO: Implement this.
    # return right_orth_lq!(A, alg_lq)
    return right_orth_svd!(A, alg_svd)
  elseif kind == :polar
    return right_orth_polar!(A, alg_polar)
  elseif kind == :svd
    return right_orth_svd!(A, alg_svd, trunc)
  else
    throw(ArgumentError("`right_orth` received unknown value `kind = $kind`"))
  end
end
function MatrixAlgebraKit.right_orth_lq!(A::AbstractBlockSparseMatrix, alg)
  alg′ = select_algorithm(lq_compact, A, alg)
  return lq_compact!(A, alg′)
end
function MatrixAlgebraKit.right_orth_polar!(A::AbstractBlockSparseMatrix, alg)
  alg′ = select_algorithm(right_polar!, A, alg)
  return right_polar!(A, alg′)
end
function MatrixAlgebraKit.right_orth_svd!(
  A::AbstractBlockSparseMatrix, alg, trunc::Nothing=nothing
)
  alg′ = select_algorithm(svd_compact!, A, alg)
  U, S, Vᴴ = svd_compact!(A, alg′)
  return U * S, Vᴴ
end
function MatrixAlgebraKit.right_orth_svd!(A::AbstractBlockSparseMatrix, alg, trunc)
  alg′ = select_algorithm(svd_compact!, A, alg)
  alg_trunc = select_algorithm(svd_trunc!, A, alg′; trunc)
  U, S, Vᴴ = svd_trunc!(A, alg_trunc)
  return U * S, Vᴴ
end
