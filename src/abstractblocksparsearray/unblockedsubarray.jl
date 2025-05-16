using ArrayLayouts: ArrayLayouts, MemoryLayout
using Base.Broadcast: Broadcast, BroadcastStyle
using BlockArrays: BlockArrays, Block, BlockIndexRange, BlockSlice
using TypeParameterAccessors: TypeParameterAccessors, parenttype, similartype

const UnblockedIndices = Union{
  Vector{<:Integer},BlockSlice{<:Block{1}},BlockSlice{<:BlockIndexRange{1}}
}

const UnblockedSubArray{T,N} = SubArray{
  T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{UnblockedIndices}}
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

function Base.similar(
  a::UnblockedSubArray, elt::Type, axes::Tuple{Base.OneTo,Vararg{Base.OneTo}}
)
  return similar(similartype(blocktype(parenttype(a)), elt), axes)
end
function Base.similar(a::UnblockedSubArray, elt::Type, size::Tuple{Int,Vararg{Int}})
  return similar(a, elt, Base.OneTo.(size))
end

function ArrayLayouts.sub_materialize(a::UnblockedSubArray)
  a_cpu = adapt(Array, a)
  a_cpu′ = similar(a_cpu)
  a_cpu′ .= a_cpu
  if typeof(a) === typeof(a_cpu)
    return a_cpu′
  end
  a′ = similar(a)
  a′ .= a_cpu′
  return a′
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

# Fix ambiguity and scalar indexing errors with GPUArrays.
using Adapt: adapt
using GPUArraysCore: GPUArraysCore
function Base.map!(
  f,
  a_dest::GPUArraysCore.AnyGPUArray,
  a_src1::UnblockedSubArray,
  a_src_rest::UnblockedSubArray...,
)
  a_dest_cpu = adapt(Array, a_dest)
  a_srcs_cpu = map(adapt(Array), (a_src1, a_src_rest...))
  map!(f, a_dest_cpu, a_srcs_cpu...)
  a_dest .= a_dest_cpu
  return a_dest
end

function Base.iszero(a::UnblockedSubArray)
  return invoke(iszero, Tuple{AbstractArray}, adapt(Array, a))
end
