using BlockSparseArrays: BlockSparseArrays
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
  # TODO: Reenable ambiguity and piracy checks.
  Aqua.test_all(BlockSparseArrays; ambiguities=false, piracies=false)
end
