using MatrixAlgebraKit:
  MatrixAlgebraKit, default_eig_algorithm, default_eigh_algorithm, eig_full!, eigh_full!

function initialize_blocksparse_eig_output(
  f, A::AbstractMatrix, alg::BlockPermutedDiagonalAlgorithm
)
  Td, Tv = fieldtypes(Base.promote_op(f, blocktype(A), typeof(alg.alg)))
  D = similar(A, BlockType(Td))
  V = similar(A, BlockType(Tv))
  return (D, V)
end

function blocksparse_eig_full!(
  f, A::AbstractMatrix, (D, V), alg::BlockPermutedDiagonalAlgorithm
)
  for I in blockdiagindices(A)
    d, v = f(@view!(A[I]), alg.alg)
    D[I], V[I] = d, v
  end
  return (D, V)
end

for f in [:default_eig_algorithm, :default_eigh_algorithm]
  @eval begin
    function MatrixAlgebraKit.$f(arrayt::Type{<:AbstractBlockSparseMatrix}; kwargs...)
      alg = $f(blocktype(arrayt); kwargs...)
      return BlockPermutedDiagonalAlgorithm(alg)
    end
  end
end

for f in [:eig_full!, :eigh_full!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(
      ::typeof($f), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
    )
      return initialize_blocksparse_eig_output($f, A, alg)
    end
    function MatrixAlgebraKit.$f(
      A::AbstractBlockSparseMatrix, (D, V), alg::BlockPermutedDiagonalAlgorithm
    )
      return blocksparse_eig_full!($f, A, (D, V), alg)
    end
  end
end
