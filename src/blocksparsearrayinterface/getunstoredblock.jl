using BlockArrays: Block
using DerivableInterfaces: zero!

struct GetUnstoredBlock{Axes}
  axes::Axes
end

# Allow customizing based on the block index.
function unstored_block(
  A::Type{<:AbstractArray{<:Any,N}}, ax::NTuple{N,AbstractUnitRange{<:Integer}}, I::Block{N}
) where {N}
  return unstored_block(A, ax)
end
function unstored_block(
  A::Type{<:AbstractArray{<:Any,N}}, ax::NTuple{N,AbstractUnitRange{<:Integer}}
) where {N}
  a = similar(A, ax)
  zero!(a)
  return a
end

using LinearAlgebra: Diagonal
# TODO: This is a hack and is also type-unstable.
function unstored_block(
  A::Type{<:Diagonal{<:Any,V}}, ax::NTuple{2,AbstractUnitRange{<:Integer}}, I::Block{2}
) where {V}
  if allequal(Tuple(I))
    # Diagonal blocks.
    diag = zero!(similar(V, first(ax)))
    return Diagonal(diag)
  else
    # Off-diagonal blocks.
    return zero!(similar(similartype(V, typeof(ax)), ax))
  end
end

@inline function (f::GetUnstoredBlock)(
  a::AbstractArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  # TODO: Make sure this works for sparse or block sparse blocks, immutable
  # blocks, diagonal blocks, etc.!
  b_ax = ntuple(ndims(a)) do d
    return only(axes(f.axes[d][Block(I[d])]))
  end
  return unstored_block(eltype(a), b_ax, Block(I))
end
# TODO: Use `Base.to_indices`.
@inline function (f::GetUnstoredBlock)(
  a::AbstractArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return f(a, Tuple(I)...)
end
