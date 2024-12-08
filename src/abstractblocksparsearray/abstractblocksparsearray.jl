using BlockArrays:
  BlockArrays, AbstractBlockArray, Block, BlockIndex, BlockedUnitRange, blocks
using SparseArraysBase: sparse_getindex, sparse_setindex!

# TODO: Delete this. This function was replaced
# by `stored_length` but is still used in `NDTensors`.
function nonzero_keys end

abstract type AbstractBlockSparseArray{T,N} <: AbstractBlockArray{T,N} end

using Derive: @array_aliases
# Define AbstractSparseVector, AnyAbstractSparseArray, etc.
@array_aliases AbstractBlockSparseArray

using Derive: Derive
function Derive.interface(::Type{<:AbstractBlockSparseArray})
  return BlockSparseArrayInterface()
end

using Derive: @derive

# TODO: These need to be loaded since `AbstractArrayOps`
# includes overloads of functions from these modules.
# Ideally that wouldn't be needed and can be circumvented
# with `GlobalRef`.
using ArrayLayouts: ArrayLayouts
using LinearAlgebra: LinearAlgebra

# Derive `Base.getindex`, `Base.setindex!`, etc.
# TODO: Define `AbstractMatrixOps` and overload for
# `AnyAbstractSparseMatrix` and `AnyAbstractSparseVector`,
# which is where matrix multiplication and factorizations
# shoudl go.
@derive AnyAbstractBlockSparseArray AbstractArrayOps

## Base `AbstractArray` interface

Base.axes(::AbstractBlockSparseArray) = error("Not implemented")

# TODO: Add some logic to unwrapping wrapped arrays.
# TODO: Decide what a good default is.
blockstype(arraytype::Type{<:AbstractBlockSparseArray}) = SparseArrayDOK{AbstractArray}
function blockstype(arraytype::Type{<:AbstractBlockSparseArray{T}}) where {T}
  return SparseArrayDOK{AbstractArray{T}}
end
function blockstype(arraytype::Type{<:AbstractBlockSparseArray{T,N}}) where {T,N}
  return SparseArrayDOK{AbstractArray{T,N},N}
end

# Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.getindex(a::AbstractBlockSparseArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return blocksparse_getindex(a, I...)
end

# Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.getindex(a::AbstractBlockSparseArray{<:Any,0})
  return blocksparse_getindex(a)
end

## # Fix ambiguity error with `BlockArrays`.
## function Base.getindex(a::AbstractBlockSparseArray{<:Any,N}, I::Block{N}) where {N}
##   return ArrayLayouts.layout_getindex(a, I)
## end
##
## # Fix ambiguity error with `BlockArrays`.
## function Base.getindex(a::AbstractBlockSparseArray{<:Any,1}, I::Block{1})
##   return ArrayLayouts.layout_getindex(a, I)
## end
##
## # Fix ambiguity error with `BlockArrays`.
## function Base.getindex(a::AbstractBlockSparseArray, I::Vararg{AbstractVector})
##   ## return blocksparse_getindex(a, I...)
##   return ArrayLayouts.layout_getindex(a, I...)
## end

# Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.setindex!(
  a::AbstractBlockSparseArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  blocksparse_setindex!(a, value, I...)
  return a
end

# Fix ambiguity error.
function Base.setindex!(a::AbstractBlockSparseArray{<:Any,0}, value)
  blocksparse_setindex!(a, value)
  return a
end

function Base.setindex!(
  a::AbstractBlockSparseArray{<:Any,N}, value, I::Vararg{Block{1},N}
) where {N}
  blocksize = ntuple(dim -> length(axes(a, dim)[I[dim]]), N)
  if size(value) ≠ blocksize
    throw(
      DimensionMismatch(
        "Trying to set block $(Block(Int.(I)...)), which has a size $blocksize, with data of size $(size(value)).",
      ),
    )
  end
  blocks(a)[Int.(I)...] = value
  return a
end
