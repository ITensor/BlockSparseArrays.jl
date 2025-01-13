using Test
using BlockSparseArrays
using BlockSparseArrays: BlockSparseArray, svd, BlockDiagonal, eachblockstoredindex
using BlockArrays
using Random
using DiagonalArrays: diagonal
using LinearAlgebra: LinearAlgebra

function test_svd(a, usv)
  U, S, V = usv

  @test U * diagonal(S) * V' ≈ a
  @test U' * U ≈ LinearAlgebra.I
  @test V' * V ≈ LinearAlgebra.I
end

# regular matrix
# --------------
sizes = ((3, 3), (4, 3), (3, 4))
eltypes = (Float32, Float64, ComplexF64)
@testset "($m, $n) Matrix{$T}" for ((m, n), T) in Iterators.product(sizes, eltypes)
  a = rand(m, n)
  usv = @inferred svd(a)
  test_svd(a, usv)
end

# block matrix
# ------------
blockszs = (([2, 2], [2, 2]), ([2, 2], [2, 3]), ([2, 2, 1], [2, 3]), ([2, 3], [2]))
@testset "($m, $n) BlockMatrix{$T}" for ((m, n), T) in Iterators.product(blockszs, eltypes)
  a = mortar([rand(T, i, j) for i in m, j in n])
  usv = svd(a)
  test_svd(a, usv)
  @test usv.U isa BlockedMatrix
  @test usv.Vt isa BlockedMatrix
  @test usv.S isa BlockedVector
end

# Block-Diagonal matrices
# -----------------------
@testset "($m, $n) BlockDiagonal{$T}" for ((m, n), T) in
                                          Iterators.product(blockszs, eltypes)
  a = BlockDiagonal([rand(T, i, j) for (i, j) in zip(m, n)])
  usv = svd(a)
  # TODO: `BlockDiagonal * Adjoint` errors
  test_svd(a, usv)
end

# blocksparse 
# -----------
@testset "($m, $n) BlockSparseMatrix{$T}" for ((m, n), T) in
                                              Iterators.product(blockszs, eltypes)
  a = BlockSparseArray{T}(m, n)

  # test empty matrix
  usv_empty = svd(a)
  test_svd(a, usv_empty)

  # test blockdiagonal
  for i in LinearAlgebra.diagind(blocks(a))
    I = CartesianIndices(blocks(a))[i]
    a[Block(I.I...)] = rand(T, size(blocks(a)[i]))
  end
  usv = svd(a)
  test_svd(a, usv)

  perm = Random.randperm(length(m))
  b = a[Block.(perm), Block.(1:length(n))]
  usv = svd(b)
  test_svd(b, usv)

  # test permuted blockdiagonal with missing row/col
  I_removed = rand(eachblockstoredindex(b))
  c = copy(b)
  delete!(blocks(c).storage, CartesianIndex(Int.(Tuple(I_removed))))
  usv = svd(c)
  test_svd(c, usv)
end
