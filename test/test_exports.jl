using BlockSparseArrays: BlockSparseArrays
using Test: @test, @testset
@testset "Test exports" begin
  exports = [
    :BlockSparseArrays,
    :BlockSparseArray,
    :BlockSparseMatrix,
    :BlockSparseVector,
    :blockstoredlength,
    :eachblockstoredindex,
    :eachstoredblock,
  ]
  @test issetequal(names(BlockSparseArrays), exports)
end
