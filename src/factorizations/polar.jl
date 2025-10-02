using MatrixAlgebraKit:
    MatrixAlgebraKit,
    PolarViaSVD,
    check_input,
    default_algorithm,
    left_polar!,
    right_polar!,
    svd_compact!

function MatrixAlgebraKit.check_input(::typeof(left_polar!), A::AbstractBlockSparseMatrix)
    @views for I in eachblockstoredindex(A)
        m, n = size(A[I])
        m >= n ||
            throw(ArgumentError("each input matrix block needs at least as many rows as columns"))
    end
    return nothing
end
function MatrixAlgebraKit.check_input(::typeof(right_polar!), A::AbstractBlockSparseMatrix)
    @views for I in eachblockstoredindex(A)
        m, n = size(A[I])
        m <= n ||
            throw(ArgumentError("each input matrix block needs at least as many columns as rows"))
    end
    return nothing
end

function MatrixAlgebraKit.left_polar!(A::AbstractBlockSparseMatrix, alg::PolarViaSVD)
    check_input(left_polar!, A)
    # TODO: Use more in-place operations here, avoid `copy`.
    U, S, Vᴴ = svd_compact!(A, alg.svdalg)
    W = U * Vᴴ
    # TODO: `copy` is required for now because of:
    # https://github.com/ITensor/BlockSparseArrays.jl/issues/24
    # Remove when that is fixed.
    P = copy(Vᴴ') * S * Vᴴ
    return (W, P)
end
function MatrixAlgebraKit.right_polar!(A::AbstractBlockSparseMatrix, alg::PolarViaSVD)
    check_input(right_polar!, A)
    # TODO: Use more in-place operations here, avoid `copy`.
    U, S, Vᴴ = svd_compact!(A, alg.svdalg)
    Wᴴ = U * Vᴴ
    # TODO: `copy` is required for now because of:
    # https://github.com/ITensor/BlockSparseArrays.jl/issues/24
    # Remove when that is fixed.
    P = U * S * copy(U')
    return (P, Wᴴ)
end
