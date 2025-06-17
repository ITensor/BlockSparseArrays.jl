using BlockArrays: Block
using DerivableInterfaces: zero!

struct GetUnstoredBlock{Axes}
  axes::Axes
end

@inline function (f::GetUnstoredBlock)(
  ::Type{<:AbstractArray{A,N}}, I::Vararg{Int,N}
) where {A,N}
  ax = ntuple(N) do d
    return only(axes(f.axes[d][Block(I[d])]))
  end
  !isconcretetype(A) && return zero!(similar(Array{eltype(A),N}, ax))
  return zero!(similar(A, ax))
end
@inline function (f::GetUnstoredBlock)(
  a::AbstractArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  return f(typeof(a), I...)
end
# TODO: Use `Base.to_indices`.
@inline function (f::GetUnstoredBlock)(
  a::AbstractArray{<:Any,N}, I::CartesianIndex{N}
) where {N}
  return f(a, Tuple(I)...)
end

# TODO: this is a hack and is also type-unstable
using LinearAlgebra: Diagonal
using TypeParameterAccessors: similartype
function (f::GetUnstoredBlock)(
  ::Type{<:AbstractMatrix{<:Diagonal{<:Any,V}}}, I::Vararg{Int,2}
) where {V}
  ax = ntuple(2) do d
    return only(axes(f.axes[d][Block(I[d])]))
  end
  if allequal(I)
    return Diagonal(zero!(similar(V, first(ax))))
  else
    return zero!(similar(similartype(V, typeof(ax)), ax))
  end
end
