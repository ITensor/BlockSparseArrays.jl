using BlockArrays: blocksizes
using DiagonalArrays: diagonal
using LinearAlgebra: LinearAlgebra, Diagonal
using MatrixAlgebraKit:
  MatrixAlgebraKit,
  TruncationStrategy,
  check_input,
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
    function MatrixAlgebraKit.$f(::Type{<:AbstractBlockSparseMatrix}; kwargs...)
      return BlockPermutedDiagonalAlgorithm() do block
        return $f(block; kwargs...)
      end
    end
  end
end

function MatrixAlgebraKit.check_input(
  ::typeof(eig_full!), A::AbstractBlockSparseMatrix, (D, V)
)
  @assert isa(D, AbstractBlockSparseMatrix) && isa(V, AbstractBlockSparseMatrix)
  @assert eltype(V) === eltype(D) === complex(eltype(A))
  @assert axes(A, 1) == axes(A, 2)
  @assert axes(A) == axes(D) == axes(V)
  return nothing
end
function MatrixAlgebraKit.check_input(
  ::typeof(eigh_full!), A::AbstractBlockSparseMatrix, (D, V)
)
  @assert isa(D, AbstractBlockSparseMatrix) && isa(V, AbstractBlockSparseMatrix)
  @assert eltype(V) === eltype(A)
  @assert eltype(D) === real(eltype(A))
  @assert axes(A, 1) == axes(A, 2)
  @assert axes(A) == axes(D) == axes(V)
  return nothing
end

function output_type(f::typeof(eig_full!), A::Type{<:AbstractMatrix{T}}) where {T}
  DV = Base.promote_op(f, A)
  !isconcretetype(DV) && return Tuple{AbstractMatrix{complex(T)},AbstractMatrix{complex(T)}}
  return DV
end
function output_type(f::typeof(eigh_full!), A::Type{<:AbstractMatrix{T}}) where {T}
  DV = Base.promote_op(f, A)
  !isconcretetype(DV) && return Tuple{AbstractMatrix{real(T)},AbstractMatrix{T}}
  return DV
end

for f in [:eig_full!, :eigh_full!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(
      ::typeof($f), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
    )
      Td, Tv = fieldtypes(output_type($f, blocktype(A)))
      D = similar(A, BlockType(Td))
      V = similar(A, BlockType(Tv))
      return (D, V)
    end
    function MatrixAlgebraKit.$f(
      A::AbstractBlockSparseMatrix, (D, V), alg::BlockPermutedDiagonalAlgorithm
    )
      check_input($f, A, (D, V))
      for I in eachstoredblockdiagindex(A)
        block = @view!(A[I])
        block_alg = block_algorithm(alg, block)
        D[I], V[I] = $f(block, block_alg)
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

function output_type(f::typeof(eig_vals!), A::Type{<:AbstractMatrix{T}}) where {T}
  D = Base.promote_op(f, A)
  !isconcretetype(D) && return AbstractVector{complex(T)}
  return D
end
function output_type(f::typeof(eigh_vals!), A::Type{<:AbstractMatrix{T}}) where {T}
  D = Base.promote_op(f, A)
  !isconcretetype(D) && return AbstractVector{real(T)}
  return D
end

for f in [:eig_vals!, :eigh_vals!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(
      ::typeof($f), A::AbstractBlockSparseMatrix, alg::BlockPermutedDiagonalAlgorithm
    )
      T = output_type($f, blocktype(A))
      return similar(A, BlockType(T), axes(A, 1))
    end
    function MatrixAlgebraKit.$f(
      A::AbstractBlockSparseMatrix, D, alg::BlockPermutedDiagonalAlgorithm
    )
      for I in eachblockstoredindex(A)
        block = @view!(A[I])
        D[I] = $f(block, block_algorithm(alg, block))
      end
      return D
    end
  end
end
