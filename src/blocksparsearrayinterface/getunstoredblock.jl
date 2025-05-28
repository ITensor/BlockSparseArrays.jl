using BlockArrays: Block
using DerivableInterfaces: zero!

struct GetUnstoredBlock{Axes}
  axes::Axes
end

@inline function (f::GetUnstoredBlock)(
  a::AbstractArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  # TODO: Make sure this works for sparse or block sparse blocks, immutable
  # blocks, diagonal blocks, etc.!
  b_ax = ntuple(ndims(a)) do d
    return only(axes(f.axes[d][Block(I[d])]))
  end
  b = similar(eltype(a), b_ax)
  zero!(b)
  return b
end
# TODO: Use `Base.to_indices`.
@inline function (f::GetUnstoredBlock)(
  a::AbstractArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return f(a, Tuple(I)...)
end

# TODO: this is a hack and is also type-unstable
function (f::GetUnstoredBlock)(
  a::AbstractMatrix{LinearAlgebra.Diagonal{T,V}}, I::Vararg{Int,2}
) where {T,V}
  b_size = ntuple(ndims(a)) do d
    return length(f.axes[d][Block(I[d])])
  end
  if I[1] == I[2]
    diag = zero!(similar(V, b_size[1]))
    return LinearAlgebra.Diagonal{T,V}(diag)
  else
    return zeros(T, b_size...)
  end
end
