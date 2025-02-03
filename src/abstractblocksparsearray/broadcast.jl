using BlockArrays: AbstractBlockedUnitRange, BlockSlice
using Base.Broadcast: Broadcast

function Broadcast.BroadcastStyle(arraytype::Type{<:AnyAbstractBlockSparseArray})
  return BlockSparseArrayStyle{ndims(arraytype)}()
end

# Fix ambiguity error with `BlockArrays`.
function Broadcast.BroadcastStyle(
  arraytype::Type{
    <:SubArray{
      <:Any,
      <:Any,
      <:AbstractBlockSparseArray,
      <:Tuple{BlockSlice{<:Any,<:Any,<:AbstractBlockedUnitRange},Vararg{Any}},
    },
  },
)
  return BlockSparseArrayStyle{ndims(arraytype)}()
end
function Broadcast.BroadcastStyle(
  arraytype::Type{
    <:SubArray{
      <:Any,
      <:Any,
      <:AbstractBlockSparseArray,
      <:Tuple{
        BlockSlice{<:Any,<:Any,<:AbstractBlockedUnitRange},
        BlockSlice{<:Any,<:Any,<:AbstractBlockedUnitRange},
        Vararg{Any},
      },
    },
  },
)
  return BlockSparseArrayStyle{ndims(arraytype)}()
end
function Broadcast.BroadcastStyle(
  arraytype::Type{
    <:SubArray{
      <:Any,
      <:Any,
      <:AbstractBlockSparseArray,
      <:Tuple{Any,BlockSlice{<:Any,<:Any,<:AbstractBlockedUnitRange},Vararg{Any}},
    },
  },
)
  return BlockSparseArrayStyle{ndims(arraytype)}()
end

# These catch cases that aren't caught by the standard
# `BlockSparseArrayStyle` definition, and also fix
# ambiguity issues.
function Base.copyto!(dest::AnyAbstractBlockSparseArray, bc::Broadcasted)
  copyto_blocksparse!(dest, bc)
  return dest
end
function Base.copyto!(
  dest::AnyAbstractBlockSparseArray, bc::Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}
)
  copyto_blocksparse!(dest, bc)
  return dest
end
function Base.copyto!(
  dest::AnyAbstractBlockSparseArray{<:Any,N}, bc::Broadcasted{BlockSparseArrayStyle{N}}
) where {N}
  copyto_blocksparse!(dest, bc)
  return dest
end
