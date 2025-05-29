using BlockArrays: Block, blocklength
using BlockSparseArrays: blockrange, eachblockaxis
using Test: @test, @testset

@testset "blockrange" begin
  r = blockrange(AbstractUnitRange{Int}[Base.OneTo(3), 1:4])
  @test eachblockaxis(r) == [Base.OneTo(3), 1:4]
  @test eachblockaxis(r)[1] === Base.OneTo(3)
  @test eachblockaxis(r)[2] === 1:4
  @test r[Block(1)] == 1:3
  @test r[Block(2)] == 4:7
  @test first(r) == 1
  @test last(r) == 7
  @test blocklength(r) == 2
  @test r == 1:7
end
