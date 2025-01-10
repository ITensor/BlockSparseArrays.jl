using ArrayLayouts: ArrayLayouts, Dot, MatMulMatAdd, MatMulVecAdd, MulAdd
using BlockArrays: BlockArrays, BlockLayout, muladd!
using DerivableInterfaces: @interface
using SparseArraysBase: SparseLayout
using LinearAlgebra: LinearAlgebra, dot, mul!

@interface ::AbstractBlockSparseArrayInterface function BlockArrays.muladd!(
  α::Number, a1::AbstractArray, a2::AbstractArray, β::Number, a_dest::AbstractArray
)
  mul!(blocks(a_dest), blocks(a1), blocks(a2), α, β)
  return a_dest
end

function ArrayLayouts.materialize!(
  m::MatMulMatAdd{
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
  },
)
  α, a1, a2, β, a_dest = m.α, m.A, m.B, m.β, m.C
  @interface BlockSparseArrayInterface() muladd!(m.α, m.A, m.B, m.β, m.C)
  return m.C
end
function ArrayLayouts.materialize!(
  m::MatMulVecAdd{
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
    <:BlockLayout{<:SparseLayout},
  },
)
  @interface BlockSparseArrayInterface() matmul!(m)
  return m.C
end

@interface ::AbstractBlockSparseArrayInterface function LinearAlgebra.dot(
  a1::AbstractArray, a2::AbstractArray
)
  # TODO: Add a check that the blocking of `a1` and `a2` are
  # the same, or the same up to a reshape.
  return dot(blocks(a1), blocks(a2))
end

function Base.copy(d::Dot{<:BlockLayout{<:SparseLayout},<:BlockLayout{<:SparseLayout}})
  return @interface BlockSparseArrayInterface() dot(d.A, d.B)
end
