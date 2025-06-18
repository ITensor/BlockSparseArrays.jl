using Adapt: adapt
using BlockArrays: Block
using BlockSparseArrays:
  BlockSparseArray, @view!, blockstoredlength, eachblockstoredindex, view!
using JLArrays: JLArray
using SparseArraysBase: isstored
using Test: @test, @testset

elts = (Float32, Float64, ComplexF32)
arrayts = (Array, JLArray)
@testset "view! (arraytype=$arrayt, eltype=$elt)" for arrayt in arrayts, elt in elts
  dev = adapt(arrayt)

  for blk in ((Block(2, 2),), (Block(2), Block(2)))
    a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
    b = view!(a, blk...)
    x = randn(elt, 3, 3)
    b .= x
    @test b == x
    @test a[blk...] == x
    @test @view(a[blk...]) == x
    @test view!(a, blk...) == x
    @test @view!(a[blk...]) == x
  end
  for blk in ((Block(2, 2),), (Block(2), Block(2)))
    a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
    b = @view! a[blk...]
    x = randn(elt, 3, 3)
    b .= x
    @test b == x
    @test a[blk...] == x
    @test @view(a[blk...]) == x
    @test view!(a, blk...) == x
    @test @view!(a[blk...]) == x
  end
  for blk in ((Block(2, 2)[2:3, 1:2],), (Block(2)[2:3], Block(2)[1:2]))
    a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
    b = view!(a, blk...)
    x = randn(elt, 2, 2)
    b .= x
    @test b == x
    @test a[blk...] == x
    @test @view(a[blk...]) == x
    @test view!(a, blk...) == x
    @test @view!(a[blk...]) == x
  end
  for blk in ((Block(2, 2)[2:3, 1:2],), (Block(2)[2:3], Block(2)[1:2]))
    a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
    b = @view! a[blk...]
    x = randn(elt, 2, 2)
    b .= x
    @test b == x
    @test a[blk...] == x
    @test @view(a[blk...]) == x
    @test view!(a, blk...) == x
    @test @view!(a[blk...]) == x
  end
  # 0-dim case
  # Regression test for https://github.com/ITensor/BlockSparseArrays.jl/issues/148
  for I in ((), (Block(),))
    a = dev(BlockSparseArray{elt}(undef))
    @test !isstored(a)
    @test iszero(blockstoredlength(a))
    @test isempty(eachblockstoredindex(a))
    @test iszero(a)
    b = @view! a[I...]
    @test isstored(a)
    @test isone(blockstoredlength(a))
    @test issetequal(eachblockstoredindex(a), [Block()])
    @test iszero(adapt(Array)(a))
    @test b isa arrayt{elt,0}
    @test size(b) == ()
    # Converting to `Array` works around a bug in `iszero(JLArray{Float64}(undef))`.
    @test iszero(adapt(Array)(b))
  end
end
