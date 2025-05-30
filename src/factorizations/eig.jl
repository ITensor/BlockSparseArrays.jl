using MatrixAlgebraKit:
  MatrixAlgebraKit, default_eig_algorithm, default_eigh_algorithm, eig_full!, eigh_full!

function MatrixAlgebraKit.default_eig_algorithm(
  arrayt::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  alg = default_eig_algorithm(blocktype(arrayt); kwargs...)
  return BlockPermutedDiagonalAlgorithm(alg)
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(eig_full!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  D = similar(A, complex(eltype(A)))
  V = similar(A, complex(eltype(A)))
  return (D, V)
end

function MatrixAlgebraKit.eig_full!(
  A::AbstractBlockSparseMatrix, (D, V), alg::BlockPermutedDiagonalAlgorithm
)
  for I in blockdiagindices(A)
    d, v = eig_full!(A[I], alg.alg)
    D[I] = d
    V[I] = v
  end
  return (D, V)
end

# TODO: this is a hardcoded for now to get around this function not being defined in the
# type domain
function MatrixAlgebraKit.default_eigh_algorithm(
  arrayt::Type{<:AbstractBlockSparseMatrix}; kwargs...
)
  alg = default_eigh_algorithm(blocktype(arrayt); kwargs...)
  return BlockPermutedDiagonalAlgorithm(alg)
end

function MatrixAlgebraKit.initialize_output(
  ::typeof(eigh_full!), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  D = similar(A, complex(eltype(A)))
  V = similar(A, complex(eltype(A)))
  return (D, V)
end

function MatrixAlgebraKit.eigh_full!(
  A::AbstractBlockSparseMatrix, (D, V), alg::BlockPermutedDiagonalAlgorithm
)
  for I in blockdiagindices(A)
    d, v = eigh_full!(A[I], alg.alg)
    D[I] = d
    V[I] = v
  end
  return (D, V)
end
