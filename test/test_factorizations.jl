using BlockArrays: Block, BlockedMatrix, BlockedVector, blocks, mortar
using BlockSparseArrays: BlockSparseArray, BlockDiagonal, eachblockstoredindex
using MatrixAlgebraKit: svd_compact, svd_full
using LinearAlgebra: LinearAlgebra
using Random: Random
using Test: @inferred, @testset, @test

function test_svd(a, (U, S, Vᴴ); full=false)
  # Check that the SVD is correct
  (U * S * Vᴴ ≈ a) || return false
  (U' * U ≈ LinearAlgebra.I) || return false
  (Vᴴ * Vᴴ' ≈ LinearAlgebra.I) || return false
  full || return true

  # Check factors are unitary
  (U * U' ≈ LinearAlgebra.I) || return false
  (Vᴴ' * Vᴴ ≈ LinearAlgebra.I) || return false
  return true
end

blockszs = (
  ([2, 2], [2, 2]), ([2, 2], [2, 3]), ([2, 2, 1], [2, 3]), ([2, 3], [2]), ([2], [2, 3])
)
eltypes = (Float32, Float64, ComplexF64)
test_params = Iterators.product(blockszs, eltypes)

# svd_compact!
# ------------
@testset "svd_compact ($m, $n) BlockSparseMatrix{$T}" for ((m, n), T) in test_params
  a = BlockSparseArray{T}(undef, m, n)

  # test empty matrix
  usv_empty = svd_compact(a)
  @test test_svd(a, usv_empty)

  # test blockdiagonal
  for i in LinearAlgebra.diagind(blocks(a))
    I = CartesianIndices(blocks(a))[i]
    a[Block(I.I...)] = rand(T, size(blocks(a)[i]))
  end
  usv = svd_compact(a)
  @test test_svd(a, usv)

  perm = Random.randperm(length(m))
  b = a[Block.(perm), Block.(1:length(n))]
  usv = svd_compact(b)
  @test test_svd(b, usv)

  # test permuted blockdiagonal with missing row/col
  I_removed = rand(eachblockstoredindex(b))
  c = copy(b)
  delete!(blocks(c).storage, CartesianIndex(Int.(Tuple(I_removed))))
  usv = svd_compact(c)
  @test test_svd(c, usv)
end

# svd_full!
# ---------
@testset "svd_full ($m, $n) BlockSparseMatrix{$T}" for ((m, n), T) in test_params
  a = BlockSparseArray{T}(undef, m, n)

  # test empty matrix
  usv_empty = svd_full(a)
  @test test_svd(a, usv_empty; full=true)

  # test blockdiagonal
  for i in LinearAlgebra.diagind(blocks(a))
    I = CartesianIndices(blocks(a))[i]
    a[Block(I.I...)] = rand(T, size(blocks(a)[i]))
  end
  usv = svd_full(a)
  @test test_svd(a, usv; full=true)

  perm = Random.randperm(length(m))
  b = a[Block.(perm), Block.(1:length(n))]
  usv = svd_full(b)
  @test test_svd(b, usv; full=true)

  # test permuted blockdiagonal with missing row/col
  I_removed = rand(eachblockstoredindex(b))
  c = copy(b)
  delete!(blocks(c).storage, CartesianIndex(Int.(Tuple(I_removed))))
  usv = svd_full(c)
  @test test_svd(c, usv; full=true)
end
