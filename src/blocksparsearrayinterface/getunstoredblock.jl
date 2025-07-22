using BlockArrays: Block
using DerivableInterfaces: zero!

struct ZeroBlocks{
  N,A<:AbstractArray{<:Any,N},ParentAxes<:Tuple{Vararg{AbstractUnitRange{<:Integer},N}}
} <: AbstractArray{A,N}
  parentaxes::ParentAxes
end
function ZeroBlocks{N,A}(
  ax::Ax
) where {N,A<:AbstractArray{<:Any,N},Ax<:Tuple{Vararg{AbstractUnitRange{<:Integer},N}}}
  return ZeroBlocks{N,A,Ax}(ax)
end
Base.size(a::ZeroBlocks) = map(blocklength, a.parentaxes)

function Base.AbstractArray{A}(a::ZeroBlocks{N}) where {N,A}
  return ZeroBlocks{N,A}(a.parentaxes)
end

@inline function Base.getindex(a::ZeroBlocks{N,A}, I::Vararg{Int,N}) where {N,A}
  # TODO: Use `BlockArrays.eachblockaxes`.
  ax = ntuple(N) do d
    return only(axes(a.parentaxes[d][Block(I[d])]))
  end
  !isconcretetype(A) && return zero!(similar(Array{eltype(A),N}, ax))
  return zero!(similar(A, ax))
end
# TODO: Use `Base.to_indices`.
@inline function Base.getindex(a::ZeroBlocks{N,A}, I::CartesianIndex{N}) where {N,A}
  return a[Tuple(I)...]
end

# TODO: this is a hack and is also type-unstable
using LinearAlgebra: Diagonal
using TypeParameterAccessors: similartype
function Base.getindex(a::ZeroBlocks{2,A}, I::Vararg{Int,2}) where {V,A<:Diagonal{<:Any,V}}
  ax = ntuple(2) do d
    return only(axes(a.parentaxes[d][Block(I[d])]))
  end
  if allequal(I)
    return Diagonal(zero!(similar(V, first(ax))))
  else
    return zero!(similar(similartype(V, typeof(ax)), ax))
  end
end
