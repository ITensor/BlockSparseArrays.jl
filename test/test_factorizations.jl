using BlockArrays: Block, BlockedMatrix, BlockedVector, blocks, mortar
using BlockSparseArrays: BlockSparseArray, BlockDiagonal, eachblockstoredindex
using MatrixAlgebraKit:
  left_polar,
  lq_compact,
  lq_full,
  qr_compact,
  qr_full,
  right_polar,
  svd_compact,
  svd_full,
  svd_trunc,
  truncrank,
  trunctol
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

# svd_trunc!
# ----------

@testset "svd_trunc ($m, $n) BlockSparseMatri{$T}" for ((m, n), T) in test_params
  a = BlockSparseArray{T}(undef, m, n)

  # test blockdiagonal
  for i in LinearAlgebra.diagind(blocks(a))
    I = CartesianIndices(blocks(a))[i]
    a[Block(I.I...)] = rand(T, size(blocks(a)[i]))
  end

  minmn = min(size(a)...)
  r = max(1, minmn - 2)
  trunc = truncrank(r)

  U1, S1, V1ᴴ = svd_trunc(a; trunc)
  U2, S2, V2ᴴ = svd_trunc(Matrix(a); trunc)
  @test size(U1) == size(U2)
  @test size(S1) == size(S2)
  @test size(V1ᴴ) == size(V2ᴴ)
  @test Matrix(U1 * S1 * V1ᴴ) ≈ U2 * S2 * V2ᴴ

  @test (U1' * U1 ≈ LinearAlgebra.I)
  @test (V1ᴴ * V1ᴴ' ≈ LinearAlgebra.I)

  atol = minimum(LinearAlgebra.diag(S1)) + 10 * eps(real(T))
  trunc = trunctol(atol)

  U1, S1, V1ᴴ = svd_trunc(a; trunc)
  U2, S2, V2ᴴ = svd_trunc(Matrix(a); trunc)
  @test size(U1) == size(U2)
  @test size(S1) == size(S2)
  @test size(V1ᴴ) == size(V2ᴴ)
  @test Matrix(U1 * S1 * V1ᴴ) ≈ U2 * S2 * V2ᴴ

  @test (U1' * U1 ≈ LinearAlgebra.I)
  @test (V1ᴴ * V1ᴴ' ≈ LinearAlgebra.I)

  # test permuted blockdiagonal
  perm = Random.randperm(length(m))
  b = a[Block.(perm), Block.(1:length(n))]
  for trunc in (truncrank(r), trunctol(atol))
    U1, S1, V1ᴴ = svd_trunc(b; trunc)
    U2, S2, V2ᴴ = svd_trunc(Matrix(b); trunc)
    @test size(U1) == size(U2)
    @test size(S1) == size(S2)
    @test size(V1ᴴ) == size(V2ᴴ)
    @test Matrix(U1 * S1 * V1ᴴ) ≈ U2 * S2 * V2ᴴ

    @test (U1' * U1 ≈ LinearAlgebra.I)
    @test (V1ᴴ * V1ᴴ' ≈ LinearAlgebra.I)
  end

  # test permuted blockdiagonal with missing row/col
  I_removed = rand(eachblockstoredindex(b))
  c = copy(b)
  delete!(blocks(c).storage, CartesianIndex(Int.(Tuple(I_removed))))
  for trunc in (truncrank(r), trunctol(atol))
    U1, S1, V1ᴴ = svd_trunc(c; trunc)
    U2, S2, V2ᴴ = svd_trunc(Matrix(c); trunc)
    @test size(U1) == size(U2)
    @test size(S1) == size(S2)
    @test size(V1ᴴ) == size(V2ᴴ)
    @test Matrix(U1 * S1 * V1ᴴ) ≈ U2 * S2 * V2ᴴ

    @test (U1' * U1 ≈ LinearAlgebra.I)
    @test (V1ᴴ * V1ᴴ' ≈ LinearAlgebra.I)
  end
end

@testset "qr_compact" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
    A = BlockSparseArray{T}(undef, ([i, j], [k, l]))
    A[Block(1, 1)] = randn(T, i, k)
    A[Block(2, 2)] = randn(T, j, l)
    Q, R = qr_compact(A)
    @test Matrix(Q'Q) ≈ LinearAlgebra.I
    @test A ≈ Q * R
  end
end

@testset "qr_full" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
    A = BlockSparseArray{T}(undef, ([i, j], [k, l]))
    A[Block(1, 1)] = randn(T, i, k)
    A[Block(2, 2)] = randn(T, j, l)
    Q, R = qr_full(A)
    Q′, R′ = qr_full(Matrix(A))
    @test size(Q) == size(Q′)
    @test size(R) == size(R′)
    @test Matrix(Q'Q) ≈ LinearAlgebra.I
    @test Matrix(Q * Q') ≈ LinearAlgebra.I
    @test A ≈ Q * R
  end
end

@testset "lq_compact" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
    A = BlockSparseArray{T}(undef, ([i, j], [k, l]))
    A[Block(1, 1)] = randn(T, i, k)
    A[Block(2, 2)] = randn(T, j, l)
    L, Q = lq_compact(A)
    @test Matrix(Q * Q') ≈ LinearAlgebra.I
    @test A ≈ L * Q
  end
end

@testset "lq_full" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
    A = BlockSparseArray{T}(undef, ([i, j], [k, l]))
    A[Block(1, 1)] = randn(T, i, k)
    A[Block(2, 2)] = randn(T, j, l)
    L, Q = lq_full(A)
    L′, Q′ = lq_full(Matrix(A))
    @test size(L) == size(L′)
    @test size(Q) == size(Q′)
    @test Matrix(Q * Q') ≈ LinearAlgebra.I
    @test Matrix(Q'Q) ≈ LinearAlgebra.I
    @test A ≈ L * Q
  end
end

@testset "left_polar (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([3, 4], [2, 3]))
  A[Block(1, 1)] = randn(T, 3, 2)
  A[Block(2, 2)] = randn(T, 4, 3)

  U, C = left_polar(A)
  @test U * C ≈ A
  @test Matrix(U'U) ≈ LinearAlgebra.I
end

@testset "right_polar (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([2, 3], [3, 4]))
  A[Block(1, 1)] = randn(T, 2, 3)
  A[Block(2, 2)] = randn(T, 3, 4)

  C, U = right_polar(A)
  @test C * U ≈ A
  @test Matrix(U * U') ≈ LinearAlgebra.I
end
