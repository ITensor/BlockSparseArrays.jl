using BlockArrays: Block, BlockedMatrix, BlockedVector, blocks, mortar
using BlockSparseArrays:
  BlockSparseArrays,
  BlockDiagonal,
  BlockSparseArray,
  BlockSparseMatrix,
  blockstoredlength,
  eachblockstoredindex
using LinearAlgebra: LinearAlgebra, Diagonal, hermitianpart, pinv
using MatrixAlgebraKit:
  diagview,
  eig_full,
  eig_trunc,
  eig_vals,
  eigh_full,
  eigh_trunc,
  eigh_vals,
  left_orth,
  left_polar,
  lq_compact,
  lq_full,
  qr_compact,
  qr_full,
  right_orth,
  right_polar,
  svd_compact,
  svd_full,
  svd_trunc,
  truncrank,
  trunctol
using Random: Random
using StableRNGs: StableRNG
using Test: @inferred, @test, @test_broken, @test_throws, @testset

@testset "Matrix functions (T=$elt)" for elt in (Float32, Float64, ComplexF64)
  rng = StableRNG(123)
  a = BlockSparseMatrix{elt}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = randn(rng, elt, 2, 2)
  a[Block(2, 2)] = randn(rng, elt, 3, 3)
  MATRIX_FUNCTIONS = BlockSparseArrays.MATRIX_FUNCTIONS
  MATRIX_FUNCTIONS = [MATRIX_FUNCTIONS; [:inv, :pinv]]
  # Only works when real, also isn't defined in Julia 1.10.
  MATRIX_FUNCTIONS = setdiff(MATRIX_FUNCTIONS, [:cbrt])
  MATRIX_FUNCTIONS_LOW_ACCURACY = [:acoth]
  for f in setdiff(MATRIX_FUNCTIONS, MATRIX_FUNCTIONS_LOW_ACCURACY)
    @eval begin
      fa = $f($a)
      @test Matrix(fa) ≈ $f(Matrix($a)) rtol = √(eps(real($elt)))
      @test fa isa BlockSparseMatrix
      @test issetequal(eachblockstoredindex(fa), [Block(1, 1), Block(2, 2)])
    end
  end
  for f in MATRIX_FUNCTIONS_LOW_ACCURACY
    @eval begin
      fa = $f($a)
      if !Sys.isapple() && ($elt <: Real)
        # `acoth` appears to be broken on this matrix on Windows and Ubuntu
        # for real matrices.
        @test_broken Matrix(fa) ≈ $f(Matrix($a)) rtol = √eps(real($elt))
      else
        @test Matrix(fa) ≈ $f(Matrix($a)) rtol = √eps(real($elt))
      end
      @test fa isa BlockSparseMatrix
      @test issetequal(eachblockstoredindex(fa), [Block(1, 1), Block(2, 2)])
    end
  end

  # Catch case of off-diagonal blocks.
  rng = StableRNG(123)
  a = BlockSparseMatrix{elt}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = randn(rng, elt, 2, 2)
  a[Block(1, 2)] = randn(rng, elt, 2, 3)
  for f in MATRIX_FUNCTIONS
    @eval begin
      @test_throws ArgumentError $f($a)
    end
  end

  # Missing diagonal blocks.
  rng = StableRNG(123)
  a = BlockSparseMatrix{elt}(undef, [2, 3], [2, 3])
  a[Block(2, 2)] = randn(rng, elt, 3, 3)
  MATRIX_FUNCTIONS = BlockSparseArrays.MATRIX_FUNCTIONS
  # These functions involve inverses so they break when there are zeros on the diagonal.
  MATRIX_FUNCTIONS_SINGULAR = [
    :log, :acsc, :asec, :acot, :acsch, :asech, :acoth, :csc, :cot, :csch, :coth
  ]
  MATRIX_FUNCTIONS = setdiff(MATRIX_FUNCTIONS, MATRIX_FUNCTIONS_SINGULAR)
  # Dense version is broken for some reason, investigate.
  MATRIX_FUNCTIONS = setdiff(MATRIX_FUNCTIONS, [:cbrt])
  for f in MATRIX_FUNCTIONS
    @eval begin
      fa = $f($a)
      @test Matrix(fa) ≈ $f(Matrix($a)) rtol = √(eps(real($elt)))
      @test fa isa BlockSparseMatrix
      @test issetequal(eachblockstoredindex(fa), [Block(1, 1), Block(2, 2)])
    end
  end

  SINGULAR_EXCEPTION = if VERSION < v"1.11-"
    # A different exception is thrown in older versions of Julia.
    LinearAlgebra.LAPACKException
  else
    LinearAlgebra.SingularException
  end
  for f in setdiff(MATRIX_FUNCTIONS_SINGULAR, [:log])
    @eval begin
      @test_throws $SINGULAR_EXCEPTION $f($a)
    end
  end
end

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

@testset "svd_trunc ($m, $n) BlockSparseMatrix{$T}" for ((m, n), T) in test_params
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

@testset "qr_compact (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
    A = BlockSparseArray{T}(undef, ([i, j], [k, l]))
    A[Block(1, 1)] = randn(T, i, k)
    A[Block(2, 2)] = randn(T, j, l)
    Q, R = qr_compact(A)
    @test Matrix(Q'Q) ≈ LinearAlgebra.I
    @test A ≈ Q * R
  end
end

@testset "qr_full (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
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

@testset "left_orth (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([3, 4], [2, 3]))
  A[Block(1, 1)] = randn(T, 3, 2)
  A[Block(2, 2)] = randn(T, 4, 3)

  for kind in (:polar, :qr, :svd)
    U, C = left_orth(A; kind)
    @test U * C ≈ A
    @test Matrix(U'U) ≈ LinearAlgebra.I
  end

  U, C = left_orth(A; trunc=(; maxrank=2))
  @test size(U, 2) ≤ 2
  @test size(C, 1) ≤ 2
  @test Matrix(U'U) ≈ LinearAlgebra.I
end

@testset "right_orth (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([2, 3], [3, 4]))
  A[Block(1, 1)] = randn(T, 2, 3)
  A[Block(2, 2)] = randn(T, 3, 4)

  for kind in (:lq, :polar, :svd)
    C, U = right_orth(A; kind)
    @test C * U ≈ A
    @test Matrix(U * U') ≈ LinearAlgebra.I
  end

  C, U = right_orth(A; trunc=(; maxrank=2))
  @test size(C, 2) ≤ 2
  @test size(U, 1) ≤ 2
  @test Matrix(U * U') ≈ LinearAlgebra.I
end

@testset "eig_full (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([2, 3], [2, 3]))
  rng = StableRNG(123)
  A[Block(1, 1)] = randn(rng, T, 2, 2)
  A[Block(2, 2)] = randn(rng, T, 3, 3)

  D, V = eig_full(A)
  @test size(D) == size(A)
  @test size(D) == size(A)
  @test blockstoredlength(D) == 2
  @test blockstoredlength(V) == 2
  @test issetequal(eachblockstoredindex(D), [Block(1, 1), Block(2, 2)])
  @test issetequal(eachblockstoredindex(V), [Block(1, 1), Block(2, 2)])
  @test A * V ≈ V * D
end

@testset "eig_vals (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([2, 3], [2, 3]))
  rng = StableRNG(123)
  A[Block(1, 1)] = randn(rng, T, 2, 2)
  A[Block(2, 2)] = randn(rng, T, 3, 3)

  D = eig_vals(A)
  @test size(D) == (size(A, 1),)
  @test blockstoredlength(D) == 2
  D′ = eig_vals(Matrix(A))
  @test sort(D; by=abs) ≈ sort(D′; by=abs)
end

@testset "eig_trunc (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([2, 3], [2, 3]))
  rng = StableRNG(123)
  D1 = [1.0, 0.1]
  V1 = randn(rng, T, 2, 2)
  A1 = V1 * Diagonal(D1) * inv(V1)
  D2 = [1.0, 0.5, 0.1]
  V2 = randn(rng, T, 3, 3)
  A2 = V2 * Diagonal(D2) * inv(V2)
  A[Block(1, 1)] = A1
  A[Block(2, 2)] = A2

  D, V = eig_trunc(A; trunc=(; maxrank=3))
  @test size(D) == (3, 3)
  @test size(D) == (3, 3)
  @test blockstoredlength(D) == 2
  @test blockstoredlength(V) == 2
  @test issetequal(eachblockstoredindex(D), [Block(1, 1), Block(2, 2)])
  @test issetequal(eachblockstoredindex(V), [Block(1, 1), Block(2, 2)])
  @test A * V ≈ V * D
  @test sort(diagview(D[Block(1, 1)]); by=abs, rev=true) ≈ D1[1:1]
  @test sort(diagview(D[Block(2, 2)]); by=abs, rev=true) ≈ D2[1:2]
end

herm(x) = parent(hermitianpart(x))

@testset "eigh_full (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([2, 3], [2, 3]))
  rng = StableRNG(123)
  A[Block(1, 1)] = herm(randn(rng, T, 2, 2))
  A[Block(2, 2)] = herm(randn(rng, T, 3, 3))

  D, V = eigh_full(A)
  @test size(D) == size(A)
  @test size(D) == size(A)
  @test blockstoredlength(D) == 2
  @test blockstoredlength(V) == 2
  @test issetequal(eachblockstoredindex(D), [Block(1, 1), Block(2, 2)])
  @test issetequal(eachblockstoredindex(V), [Block(1, 1), Block(2, 2)])
  @test A * V ≈ V * D
end

@testset "eigh_vals (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([2, 3], [2, 3]))
  rng = StableRNG(123)
  A[Block(1, 1)] = herm(randn(rng, T, 2, 2))
  A[Block(2, 2)] = herm(randn(rng, T, 3, 3))

  D = eigh_vals(A)
  @test size(D) == (size(A, 1),)
  @test blockstoredlength(D) == 2
  D′ = eigh_vals(Matrix(A))
  @test sort(D; by=abs) ≈ sort(D′; by=abs)
end

@testset "eigh_trunc (T=$T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
  A = BlockSparseArray{T}(undef, ([2, 3], [2, 3]))
  rng = StableRNG(123)
  D1 = [1.0, 0.1]
  V1, _ = qr_compact(randn(rng, T, 2, 2))
  A1 = V1 * Diagonal(D1) * V1'
  D2 = [1.0, 0.5, 0.1]
  V2, _ = qr_compact(randn(rng, T, 3, 3))
  A2 = V2 * Diagonal(D2) * V2'
  A[Block(1, 1)] = herm(A1)
  A[Block(2, 2)] = herm(A2)

  D, V = eigh_trunc(A; trunc=(; maxrank=3))
  @test size(D) == (3, 3)
  @test size(D) == (3, 3)
  @test blockstoredlength(D) == 2
  @test blockstoredlength(V) == 2
  @test issetequal(eachblockstoredindex(D), [Block(1, 1), Block(2, 2)])
  @test issetequal(eachblockstoredindex(V), [Block(1, 1), Block(2, 2)])
  @test A * V ≈ V * D
  @test sort(diagview(D[Block(1, 1)]); by=abs, rev=true) ≈ D1[1:1]
  @test sort(diagview(D[Block(2, 2)]); by=abs, rev=true) ≈ D2[1:2]
end
