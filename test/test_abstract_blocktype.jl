using Adapt: adapt
using BlockArrays: Block
using BlockSparseArrays: BlockSparseMatrix, blockstoredlength
using JLArrays: JLArray
using LinearAlgebra: hermitianpart, norm
using MatrixAlgebraKit:
  eig_full,
  eig_trunc,
  eig_vals,
  eigh_full,
  eigh_trunc,
  eigh_vals,
  isisometry,
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
  svd_trunc
using SparseArraysBase: storedlength
using Test: @test, @test_broken, @testset

elts = (Float32, Float64, ComplexF32)
arrayts = (Array, JLArray)
@testset "Abstract block type (arraytype=$arrayt, eltype=$elt)" for arrayt in arrayts,
  elt in elts

  dev = adapt(arrayt)

  a = BlockSparseMatrix{elt,AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
  @test sprint(show, MIME"text/plain"(), a) isa String
  @test iszero(storedlength(a))
  @test iszero(blockstoredlength(a))

  a = BlockSparseMatrix{elt,AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = dev(randn(elt, 2, 2))
  @test !iszero(a[Block(1, 1)])
  @test a[Block(1, 1)] isa arrayt{elt,2}
  @test iszero(a[Block(2, 2)])
  @test a[Block(2, 2)] isa Matrix{elt}
  @test iszero(a[Block(2, 1)])
  @test a[Block(2, 1)] isa Matrix{elt}
  @test iszero(a[Block(1, 2)])
  @test a[Block(1, 2)] isa Matrix{elt}

  a = BlockSparseMatrix{elt,AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = dev(randn(elt, 2, 2))
  a′ = BlockSparseMatrix{elt,AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
  a′[Block(2, 2)] = dev(randn(elt, 3, 3))

  b = copy(a)
  @test Array(b) ≈ Array(a)

  b = a + a′
  @test Array(b) ≈ Array(a) + Array(a′)

  b = 3a
  @test Array(b) ≈ 3Array(a)

  b = a * a
  @test Array(b) ≈ Array(a) * Array(a)

  b = a * a′
  @test Array(b) ≈ Array(a) * Array(a′)
  @test norm(b) ≈ 0

  a = BlockSparseMatrix{elt,AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = dev(randn(elt, 2, 2))
  for f in (eig_full, eig_trunc)
    if arrayt === Array
      d, v = f(a)
      @test a * v ≈ v * d
    else
      @test_broken f(a)
    end
  end
  if arrayt === Array
    d = eig_vals(a)
    @test sort(Vector(d); by=abs) ≈ sort(eig_vals(Matrix(a)); by=abs)
  else
    @test_broken eig_vals(a)
  end

  a = BlockSparseMatrix{elt,AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = dev(parent(hermitianpart(randn(elt, 2, 2))))
  for f in (eigh_full, eigh_trunc)
    if arrayt === Array
      d, v = f(a)
      @test a * v ≈ v * d
    else
      @test_broken f(a)
    end
  end
  if arrayt === Array
    d = eigh_vals(a)
    @test sort(Vector(d); by=abs) ≈ sort(eig_vals(Matrix(a)); by=abs)
  else
    @test_broken eigh_vals(a)
  end

  a = BlockSparseMatrix{elt,AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = dev(randn(elt, 2, 2))
  for f in (left_orth, left_polar, qr_compact, qr_full)
    if arrayt === Array
      u, c = f(a)
      @test u * c ≈ a
      @test isisometry(u; side=:left)
    else
      @test_broken f(a)
    end
  end
  for f in (right_orth, right_polar, lq_compact, lq_full)
    if arrayt === Array
      c, u = f(a)
      @test c * u ≈ a
      @test isisometry(u; side=:right)
    else
      @test_broken f(a)
    end
  end
  for f in (svd_compact, svd_full, svd_trunc)
    if arrayt === Array
      u, s, v = f(a)
      @test u * s * v ≈ a
    else
      @test_broken f(a)
    end
  end
end
