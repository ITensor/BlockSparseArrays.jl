using ArrayLayouts: zero!
using BlockArrays: Block

struct GetUnstoredBlock{Axes}
  axes::Axes
end

@inline function (f::GetUnstoredBlock)(
  a::AbstractArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  # TODO: Make sure this works for sparse or block sparse blocks, immutable
  # blocks, diagonal blocks, etc.!
  b_size = ntuple(ndims(a)) do d
    return length(f.axes[d][Block(I[d])])
  end
  b = similar(eltype(a), b_size)
  zero!(b)
  return b
end
# TODO: Use `Base.to_indices`.
@inline function (f::GetUnstoredBlock)(
  a::AbstractArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return f(a, Tuple(I)...)
end
