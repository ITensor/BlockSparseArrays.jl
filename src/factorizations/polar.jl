using MatrixAlgebraKit:
  MatrixAlgebraKit,
  PolarViaSVD,
  check_input,
  default_algorithm,
  left_polar!,
  right_polar!,
  svd_compact!

function MatrixAlgebraKit.check_input(
  ::typeof(left_polar!), A::AbstractBlockSparseMatrix, WP
)
  W, P = WP
  @views for I in eachblockstoredindex(A)
    m, n = size(A[I])
    m >= n ||
      throw(ArgumentError("each input matrix block needs at least as many rows as columns"))
    # check_input(left_polar!, A[I], (W[I1], P[I2]))
  end
  return nothing
end

function MatrixAlgebraKit.left_polar!(A::AbstractBlockSparseMatrix, WP, alg::PolarViaSVD)
  check_input(left_polar!, A, WP)
  U, S, Vᴴ = svd_compact!(A, alg.svdalg)
  # TODO: Use more in-place operations here, avoid `copy`.
  W = U * Vᴴ
  P = copy(Vᴴ') * S * Vᴴ
  return (W, P)
end
function MatrixAlgebraKit.right_polar!(A::AbstractBlockSparseMatrix, PWᴴ, alg::PolarViaSVD)
  check_input(right_polar!, A, PWᴴ)
  U, S, Vᴴ = svd_compact!(A, alg.svdalg)
  # TODO: Use more in-place operations here, avoid `copy`.
  Wᴴ = U * Vᴴ
  P = U * S * copy(U')
  return (P, Wᴴ)
end

function MatrixAlgebraKit.default_algorithm(
  ::typeof(left_polar!), a::AbstractBlockSparseMatrix; kwargs...
)
  return PolarViaSVD(default_algorithm(svd_compact!, a; kwargs...))
end
function MatrixAlgebraKit.default_algorithm(
  ::typeof(right_polar!), a::AbstractBlockSparseMatrix; kwargs...
)
  return PolarViaSVD(default_algorithm(svd_compact!, a; kwargs...))
end
