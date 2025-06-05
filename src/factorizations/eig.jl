using BlockArrays: blocksizes
using DiagonalArrays: diagonal
using LinearAlgebra: LinearAlgebra, Diagonal
using MatrixAlgebraKit:
  MatrixAlgebraKit,
  TruncationStrategy,
  default_eig_algorithm,
  default_eigh_algorithm,
  diagview,
  eig_full!,
  eig_trunc!,
  eig_vals!,
  eigh_full!,
  eigh_trunc!,
  eigh_vals!,
  findtruncated

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
      Td, Tv = fieldtypes(Base.promote_op($f, blocktype(A), typeof(alg.alg)))
      D = similar(A, BlockType(Td))
      V = similar(A, BlockType(Tv))
      return (D, V)
    end
    function MatrixAlgebraKit.$f(
      A::AbstractBlockSparseMatrix, (D, V), alg::BlockPermutedDiagonalAlgorithm
    )
      for I in eachstoredblockdiagindex(A)
        D[I], V[I] = $f(@view(A[I]), alg.alg)
      end
      for I in eachunstoredblockdiagindex(A)
        # TODO: Support setting `LinearAlgebra.I` directly, and/or
        # using `FillArrays.Eye`.
        V[I] = LinearAlgebra.I(size(@view(V[I]), 1))
      end
      return (D, V)
    end
  end
end

for f in [:eig_vals!, :eigh_vals!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(
      ::typeof($f), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
    )
      T = Base.promote_op($f, blocktype(A), typeof(alg.alg))
      return similar(A, BlockType(T), axes(A, 1))
    end
    function MatrixAlgebraKit.$f(
      A::AbstractBlockSparseMatrix, D, alg::BlockPermutedDiagonalAlgorithm
    )
      for I in eachblockstoredindex(A)
        D[I] = $f(@view!(A[I]), alg.alg)
      end
      return D
    end
  end
end
