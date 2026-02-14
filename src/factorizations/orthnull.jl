using MatrixAlgebraKit: MatrixAlgebraKit, AbstractAlgorithm, LeftOrthAlgorithm,
    default_svd_algorithm, left_null!, left_orth!, left_polar!, lq_compact!,
    null_truncation_strategy, qr_compact!, right_null!, right_orth!, right_polar!,
    select_algorithm, svd_compact!
for kind in ("polar", "qr", "svd")
    @eval begin
        function MatrixAlgebraKit.initialize_output(
                ::typeof(left_orth!), A::AbstractBlockSparseMatrix,
                alg::LeftOrthAlgorithm{Symbol($kind)}
            )
            return nothing
        end
    end
end
function MatrixAlgebraKit.check_input(
        ::typeof(left_orth!), A::AbstractBlockSparseMatrix, F, alg::AbstractAlgorithm
    )
    !isnothing(F) && throw(
        ArgumentError(
            "`left_orth!` on block sparse matrices does not support specifying the output"
        )
    )
    return nothing
end

using MatrixAlgebraKit: LeftOrthViaQR
function MatrixAlgebraKit.left_orth!(A::AbstractBlockSparseMatrix, F, alg::LeftOrthViaQR)
    !isnothing(F) && throw(
        ArgumentError(
            "`left_orth!` on block sparse matrices does not support specifying the output"
        )
    )
    alg′ = select_algorithm(qr_compact!, A, alg.alg)
    return qr_compact!(A, alg′)
end
using MatrixAlgebraKit: LeftOrthViaPolar
function MatrixAlgebraKit.left_orth!(A::AbstractBlockSparseMatrix, F, alg::LeftOrthViaPolar)
    !isnothing(F) && throw(
        ArgumentError(
            "`left_orth!` on block sparse matrices does not support specifying the output"
        )
    )
    alg′ = select_algorithm(left_polar!, A, alg.alg)
    return left_polar!(A, alg′)
end
using MatrixAlgebraKit: LeftOrthViaSVD, does_truncate
function MatrixAlgebraKit.left_orth!(
        A::AbstractBlockSparseMatrix, F, alg::LeftOrthViaSVD
    )
    !isnothing(F) && throw(
        ArgumentError(
            "`left_orth!` on block sparse matrices does not support specifying the output"
        )
    )
    U, S, Vᴴ = if !does_truncate(alg.alg)
        alg′ = select_algorithm(svd_compact!, A, alg.alg)
        svd_compact!(A, alg′)
    else
        alg′ = select_algorithm(svd_compact!, A, alg.alg)
        alg_trunc = select_algorithm(svd_trunc!, A, alg′)
        svd_trunc!(A, alg_trunc)
    end
    return U, S * Vᴴ
end

using MatrixAlgebraKit: RightOrthAlgorithm
for kind in ("lq", "polar", "svd")
    @eval begin
        function MatrixAlgebraKit.initialize_output(
                ::typeof(right_orth!), A::AbstractBlockSparseMatrix,
                alg::RightOrthAlgorithm{Symbol($kind)}
            )
            return nothing
        end
    end
end
function MatrixAlgebraKit.check_input(
        ::typeof(right_orth!), A::AbstractBlockSparseMatrix, F::Nothing,
        alg::AbstractAlgorithm
    )
    !isnothing(F) && throw(
        ArgumentError(
            "`right_orth!` on block sparse matrices does not support specifying the output"
        )
    )
    return nothing
end

using MatrixAlgebraKit: RightOrthViaLQ
function MatrixAlgebraKit.right_orth!(A::AbstractBlockSparseMatrix, F, alg::RightOrthViaLQ)
    !isnothing(F) && throw(
        ArgumentError(
            "`right_orth!` on block sparse matrices does not support specifying the output"
        )
    )
    alg′ = select_algorithm(lq_compact!, A, alg.alg)
    return lq_compact!(A, alg′)
end
using MatrixAlgebraKit: RightOrthViaPolar
function MatrixAlgebraKit.right_orth!(
        A::AbstractBlockSparseMatrix, F, alg::RightOrthViaPolar
    )
    !isnothing(F) && throw(
        ArgumentError(
            "`right_orth!` on block sparse matrices does not support specifying the output"
        )
    )
    alg′ = select_algorithm(right_polar!, A, alg.alg)
    return right_polar!(A, alg′)
end
using MatrixAlgebraKit: RightOrthViaSVD
function MatrixAlgebraKit.right_orth!(
        A::AbstractBlockSparseMatrix, F, alg::RightOrthViaSVD
    )
    !isnothing(F) && throw(
        ArgumentError(
            "`right_orth!` on block sparse matrices does not support specifying the output"
        )
    )
    U, S, Vᴴ = if !does_truncate(alg.alg)
        alg′ = select_algorithm(svd_compact!, A, alg.alg)
        svd_compact!(A, alg′)
    else
        alg′ = select_algorithm(svd_compact!, A, alg.alg)
        alg_trunc = select_algorithm(svd_trunc!, A, alg′)
        svd_trunc!(A, alg_trunc)
    end
    return U * S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(
        ::typeof(left_null!), A::AbstractBlockSparseMatrix, alg::AbstractAlgorithm
    )
    return nothing
end
function MatrixAlgebraKit.check_input(
        ::typeof(left_null!), A::AbstractBlockSparseMatrix, N::Nothing,
        alg::AbstractAlgorithm
    )
    return nothing
end
function MatrixAlgebraKit.left_null!(
        A::AbstractBlockSparseMatrix, N, alg::AbstractAlgorithm
    )
    return error("Not implemented.")
end
function MatrixAlgebraKit.truncate(
        ::typeof(left_null!),
        (U, S)::Tuple{AbstractBlockSparseMatrix, AbstractBlockSparseMatrix},
        strategy::TruncationStrategy
    )
    return error("Not implemented.")
end

function MatrixAlgebraKit.initialize_output(
        ::typeof(right_null!), A::AbstractBlockSparseMatrix, alg::AbstractAlgorithm
    )
    return nothing
end
function MatrixAlgebraKit.check_input(
        ::typeof(right_null!), A::AbstractBlockSparseMatrix, N::Nothing,
        alg::AbstractAlgorithm
    )
    return nothing
end
function MatrixAlgebraKit.right_null!(
        A::AbstractBlockSparseMatrix, N, alg::AbstractAlgorithm
    )
    return error("Not implement.")
end
function MatrixAlgebraKit.truncate(
        ::typeof(right_null!),
        (S, Vᴴ)::Tuple{AbstractBlockSparseMatrix, AbstractBlockSparseMatrix},
        strategy::TruncationStrategy
    )
    return error("Not implemented.")
end
