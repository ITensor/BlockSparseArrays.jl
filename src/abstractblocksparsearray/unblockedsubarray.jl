using ArrayLayouts: ArrayLayouts, MemoryLayout
using Base.Broadcast: Broadcast, BroadcastStyle
using BlockArrays: BlockArrays
using TypeParameterAccessors: TypeParameterAccessors, parenttype, similartype

const UnblockedSubArray{T,N} = SubArray{
  T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{Vector{<:Integer}}}
}

function BlockArrays.blocks(a::UnblockedSubArray)
  return SingleBlockView(a)
end

function DerivableInterfaces.interface(arraytype::Type{<:UnblockedSubArray})
  return interface(blocktype(parenttype(arraytype)))
end

function ArrayLayouts.MemoryLayout(arraytype::Type{<:UnblockedSubArray})
  return MemoryLayout(blocktype(parenttype(arraytype)))
end

function Broadcast.BroadcastStyle(arraytype::Type{<:UnblockedSubArray})
  return BroadcastStyle(blocktype(parenttype(arraytype)))
end

function TypeParameterAccessors.similartype(arraytype::Type{<:UnblockedSubArray}, elt::Type)
  return similartype(blocktype(parenttype(arraytype)), elt)
end

function Base.map!(
  f, a_dest::AbstractArray, a_src1::UnblockedSubArray, a_src_rest::UnblockedSubArray...
)
  return invoke(
    map!,
    Tuple{Any,AbstractArray,AbstractArray,Vararg{AbstractArray}},
    f,
    a_dest,
    a_src1,
    a_src_rest...,
  )
end
function Base.iszero(a::UnblockedSubArray)
  return invoke(iszero, Tuple{AbstractArray}, a)
end
