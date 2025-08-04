using BlockArrays
using BlockSparseArrays
using BlockSparseArrays: blocksparse
using MatrixAlgebraKit
using LinearAlgebra: LinearAlgebra
using Test: @test, @testset

@testset "Issue 162" begin
  ax = (blockedrange([2, 2]), blockedrange([2, 2, 2]))
  bs = Dict(Block(1, 1) => randn(2, 2), Block(2, 3) => randn(2, 2))
  a = blocksparse(bs, ax)
  U, S, Vᴴ = svd_compact(a)

  @test U * S * Vᴴ ≈ a
  @test U' * U ≈ LinearAlgebra.I
  @test Vᴴ * Vᴴ' ≈ LinearAlgebra.I

  U, S, Vᴴ = svd_full(a);

  @test U * S * Vᴴ ≈ a
  @test U' * U ≈ LinearAlgebra.I
  @test U * U' ≈ LinearAlgebra.I
  @test Vᴴ * Vᴴ' ≈ LinearAlgebra.I
  @test Vᴴ' * Vᴴ ≈ LinearAlgebra.I
end

