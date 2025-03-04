using BlockArrays: BlockArrays, Block, BlockedUnitRange, blockedrange, blocklength
using DerivableInterfaces: @interface
using Dictionaries: Dictionary
using SparseArraysBase: SparseArrayDOK

function _BlockSparseArray end

struct BlockSparseArray{
  T,
  N,
  A<:AbstractArray{T,N},
  Blocks<:AbstractArray{A,N},
  Axes<:Tuple{Vararg{AbstractUnitRange{<:Integer},N}},
} <: AbstractBlockSparseArray{T,N}
  blocks::Blocks
  axes::Axes
  global @inline function _BlockSparseArray(
    blocks::AbstractArray{<:AbstractArray{T,N},N},
    axes::Tuple{Vararg{AbstractUnitRange{<:Integer},N}},
  ) where {T,N}
    Base.require_one_based_indexing(axes...)
    Base.require_one_based_indexing(blocks)
    return new{T,N,eltype(blocks),typeof(blocks),typeof(axes)}(blocks, axes)
  end
end

# TODO: Can this definition be shortened?
const BlockSparseMatrix{T,A<:AbstractMatrix{T},Blocks<:AbstractMatrix{A},Axes<:Tuple{AbstractUnitRange{<:Integer},AbstractUnitRange{<:Integer}}} = BlockSparseArray{
  T,2,A,Blocks,Axes
}

# TODO: Can this definition be shortened?
const BlockSparseVector{T,A<:AbstractVector{T},Blocks<:AbstractVector{A},Axes<:Tuple{AbstractUnitRange{<:Integer}}} = BlockSparseArray{
  T,1,A,Blocks,Axes
}

"""
    sparsemortar(blocks::AbstractArray{<:AbstractArray{T,N},N}, axes) -> ::BlockSparseArray{T,N}

Construct a block sparse array from a sparse array of arrays and specified blocked axes.
The block sizes must be commensurate with the blocks of the axes.
"""
function sparsemortar(
  blocks::AbstractArray{<:AbstractArray{T,N},N},
  axes::Tuple{Vararg{AbstractUnitRange{<:Integer},N}},
) where {T,N}
  return _BlockSparseArray(blocks, axes)
end

function BlockArrays.mortar(
  blocks::SparseArrayDOK{<:AbstractArray{T,N},N},
  axes::Tuple{Vararg{AbstractUnitRange{<:Integer},N}},
) where {T,N}
  return _BlockSparseArray(blocks, axes)
end

@doc """
    BlockSparseArray{T}(undef, dims)
    BlockSparseArray{T,N}(undef, dims)
    BlockSparseArray{T,N,A}(undef, dims)

Construct an uninitialized N-dimensional BlockSparseArray containing elements of type T. `dims` should be a list
of block lengths in each dimension or a list of blocked ranges representing the axes.
""" BlockSparseArray

function BlockSparseArray{T,N,A}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange,N}}
) where {T,N,A<:AbstractArray{T,N}}
  blocks = default_blocks(A, axes)
  return _BlockSparseArray(blocks, axes)
end

function BlockSparseArray{T,N,A}(
  ::UndefInitializer, axes::Vararg{AbstractUnitRange,N}
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray{T,N,A}(undef, axes)
end

function BlockSparseArray{T,N,A}(
  ::UndefInitializer, dims::Tuple{Vararg{Vector{Int},N}}
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray{T,N,A}(undef, blockedrange.(dims))
end

# Fix ambiguity error.
function BlockSparseArray{T,0,A}(
  ::UndefInitializer, axes::Tuple{}
) where {T,A<:AbstractArray{T,0}}
  blocks = default_blocks(A, axes)
  return _BlockSparseArray(blocks, axes)
end

function BlockSparseArray{T,N,A}(
  ::UndefInitializer, dims::Vararg{Vector{Int},N}
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray{T,N,A}(undef, dims)
end

function BlockSparseArray{T,N}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange,N}}
) where {T,N}
  return BlockSparseArray{T,N,default_arraytype(T, axes)}(undef, axes)
end

function BlockSparseArray{T,N}(
  ::UndefInitializer, axes::Vararg{AbstractUnitRange,N}
) where {T,N}
  return BlockSparseArray{T,N}(undef, axes)
end

function BlockSparseArray{T,0}(::UndefInitializer, axes::Tuple{}) where {T}
  return BlockSparseArray{T,0,default_arraytype(T, axes)}(undef, axes)
end

function BlockSparseArray{T,N}(
  ::UndefInitializer, dims::Tuple{Vararg{Vector{Int},N}}
) where {T,N}
  return BlockSparseArray{T,N}(undef, blockedrange.(dims))
end

function BlockSparseArray{T,N}(::UndefInitializer, dims::Vararg{Vector{Int},N}) where {T,N}
  return BlockSparseArray{T,N}(undef, dims)
end

function BlockSparseArray{T}(::UndefInitializer, dims::Tuple{Vararg{Vector{Int}}}) where {T}
  return BlockSparseArray{T,length(dims)}(undef, dims)
end

function BlockSparseArray{T}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange}}
) where {T}
  return BlockSparseArray{T,length(axes)}(undef, axes)
end

function BlockSparseArray{T}(::UndefInitializer, axes::Tuple{}) where {T}
  return BlockSparseArray{T,length(axes)}(undef, axes)
end

function BlockSparseArray{T}(::UndefInitializer, dims::Vararg{Vector{Int}}) where {T}
  return BlockSparseArray{T}(undef, dims)
end

function BlockSparseArray{T}(::UndefInitializer, axes::Vararg{AbstractUnitRange}) where {T}
  return BlockSparseArray{T}(undef, axes)
end

function BlockSparseArray{T}(::UndefInitializer) where {T}
  return BlockSparseArray{T}(undef, ())
end

# Base `AbstractArray` interface
Base.axes(a::BlockSparseArray) = a.axes

# BlockArrays `AbstractBlockArray` interface.
# This is used by `blocks(::AnyAbstractBlockSparseArray)`.
@interface ::AbstractBlockSparseArrayInterface BlockArrays.blocks(a::BlockSparseArray) =
  a.blocks

# TODO: Use `TypeParameterAccessors`.
function blockstype(
  arraytype::Type{<:BlockSparseArray{T,N,A,Blocks}}
) where {T,N,A<:AbstractArray{T,N},Blocks<:AbstractArray{A,N}}
  return Blocks
end
function blockstype(
  arraytype::Type{<:BlockSparseArray{T,N,A}}
) where {T,N,A<:AbstractArray{T,N}}
  return SparseArrayDOK{A,N}
end
function blockstype(arraytype::Type{<:BlockSparseArray{T,N}}) where {T,N}
  return SparseArrayDOK{AbstractArray{T,N},N}
end
function blockstype(arraytype::Type{<:BlockSparseArray{T}}) where {T}
  return SparseArrayDOK{AbstractArray{T}}
end
blockstype(arraytype::Type{<:BlockSparseArray}) = SparseArrayDOK{AbstractArray}

## # Base interface
## function Base.similar(
##   a::AbstractBlockSparseArray, elt::Type, axes::Tuple{Vararg{BlockedUnitRange}}
## )
##   # TODO: Preserve GPU data!
##   return BlockSparseArray{elt}(undef, axes)
## end

# TypeParameterAccessors.jl interface
using TypeParameterAccessors: TypeParameterAccessors, Position, set_type_parameters
TypeParameterAccessors.position(::Type{BlockSparseArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{BlockSparseArray}, ::typeof(ndims)) = Position(2)
TypeParameterAccessors.position(::Type{BlockSparseArray}, ::typeof(blocktype)) = Position(3)
function TypeParameterAccessors.position(::Type{BlockSparseArray}, ::typeof(blockstype))
  return Position(4)
end
