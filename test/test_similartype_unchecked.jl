using BlockSparseArrays: similartype_unchecked
using JLArrays: JLArray, JLMatrix
using Test: @test, @testset
using TestExtras: @constinferred

@testset "similartype_unchecked" begin
  @test @constinferred(similartype_unchecked(Array{Float32}, NTuple{2,Int})) ===
    Matrix{Float32}
  @test @constinferred(similartype_unchecked(Array{Float32}, NTuple{2,Base.OneTo{Int}})) ===
    Matrix{Float32}
  if VERSION < v"1.11-"
    # Not type stable in Julia 1.10.
    @test similartype_unchecked(AbstractArray{Float32}, NTuple{2,Int}) === Matrix{Float32}
    @test similartype_unchecked(JLArray{Float32}, NTuple{2,Int}) === JLMatrix{Float32}
    @test similartype_unchecked(JLArray{Float32}, NTuple{2,Base.OneTo{Int}}) ===
      JLMatrix{Float32}
  else
    @test @constinferred(similartype_unchecked(AbstractArray{Float32}, NTuple{2,Int})) ===
      Matrix{Float32}
    @test @constinferred(similartype_unchecked(JLArray{Float32}, NTuple{2,Int})) ===
      JLMatrix{Float32}
    @test @constinferred(
      similartype_unchecked(JLArray{Float32}, NTuple{2,Base.OneTo{Int}})
    ) === JLMatrix{Float32}
  end
end
