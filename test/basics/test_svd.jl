using Test
using BlockSparseArrays
using BlockSparseArrays: BlockSparseArray, svd, BlockDiagonal
using BlockArrays
using LinearAlgebra: LinearAlgebra, Diagonal, svdvals
using Random

function test_svd(a, usv)
  U, S, V = usv

  @test U * Diagonal(S) * V' ≈ a
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
  @test usv.U isa BlockDiagonal
  @test usv.Vt isa BlockDiagonal
  @test usv.S isa BlockVector
end

a = mortar([rand(2, 2) for i in 1:2, j in 1:3])
usv = svd(a)
test_svd(a, usv)

a = mortar([rand(2, 2) for i in 1:3, j in 1:2])
usv = svd(a)
test_svd(a, usv)

# blocksparse 
# -----------
@testset "($m, $n) BlockSparseMatrix{$T}" for ((m, n), T) in
                                              Iterators.product(blockszs, eltypes)
  a = BlockSparseArray{T}(m, n)
  for i in LinearAlgebra.diagind(blocks(a))
    I = CartesianIndices(blocks(a))[i]
    a[Block(I.I...)] = rand(T, size(blocks(a)[i]))
  end
  perm = Random.randperm(length(m))
  a = a[Block.(perm), Block.(1:length(n))]

  # errors because `blocks(a)[CartesianIndex.(...)]` is not implemented
  usv = svd(a)
  # TODO: `BlockDiagonal * Adjoint` errors
  test_svd(a, usv)
end
