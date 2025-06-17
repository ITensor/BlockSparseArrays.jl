using Adapt: adapt
using BlockArrays: Block
using BlockSparseArrays: BlockSparseMatrix, blockstoredlength
using JLArrays: JLArray
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
  a[Block(1, 1)] = dev(randn(elt, 2, 2))
  a[Block(2, 2)] = dev(randn(elt, 3, 3))
  @test !iszero(a[Block(1, 1)])
  @test a[Block(1, 1)] isa arrayt{elt,2}
  @test !iszero(a[Block(2, 2)])
  @test a[Block(2, 2)] isa arrayt{elt,2}
  @test iszero(a[Block(2, 1)])
  @test a[Block(2, 1)] isa Matrix{elt}
  @test iszero(a[Block(1, 2)])
  @test a[Block(1, 2)] isa Matrix{elt}

  b = copy(a)
  @test Array(b) ≈ Array(a)

  b = a + a
  @test Array(b) ≈ Array(a) + Array(a)

  b = 3a
  @test Array(b) ≈ 3Array(a)

  if arrayt === Array
    b = a * a
    @test Array(b) ≈ Array(a) * Array(a)
  else
    @test_broken a * a
  end
end
