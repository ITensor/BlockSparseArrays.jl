using BlockArrays:
  BlockArrays,
  Block,
  BlockedUnitRange,
  UndefBlocksInitializer,
  blockedrange,
  blocklength,
  undef_blocks
using DerivableInterfaces: @interface
using Dictionaries: Dictionary
using SparseArraysBase: SparseArrayDOK
using TypeParameterAccessors: similartype

"""
    SparseArrayDOK{T}(undef_blocks, axes)
    SparseArrayDOK{T,N}(undef_blocks, axes)

Construct the block structure of an undefined BlockSparseArray that will have
blocked axes `axes`.

Note that `undef_blocks` is defined in
[BlockArrays.jl](https://juliaarrays.github.io/BlockArrays.jl/stable/lib/public/#BlockArrays.undef_blocks)
and should be imported from that package to use it as an input to this constructor.
"""
function SparseArraysBase.SparseArrayDOK{T,N}(
  ::UndefBlocksInitializer, ax::Tuple{Vararg{AbstractUnitRange{<:Integer},N}}
) where {T,N}
  return SparseArrayDOK{T,N}(undef, blocklength.(ax); getunstored=GetUnstoredBlock(ax))
end
function SparseArraysBase.SparseArrayDOK{T,N}(
  ::UndefBlocksInitializer, ax::Vararg{AbstractUnitRange{<:Integer},N}
) where {T,N}
  return SparseArrayDOK{T,N}(undef_blocks, ax)
end
function SparseArraysBase.SparseArrayDOK{T,N}(
  ::UndefBlocksInitializer,
  dims::Tuple{AbstractVector{<:Integer},Vararg{AbstractVector{<:Integer}}},
) where {T,N}
  return SparseArrayDOK{T,N}(undef_blocks, blockedrange.(dims))
end
function SparseArraysBase.SparseArrayDOK{T,N}(
  ::UndefBlocksInitializer,
  dim1::AbstractVector{<:Integer},
  dim_rest::AbstractVector{<:Integer}...,
) where {T,N}
  return SparseArrayDOK{T,N}(undef_blocks, (dim1, dim_rest...))
end

function SparseArraysBase.SparseArrayDOK{T}(
  ::UndefBlocksInitializer, ax::Tuple{Vararg{AbstractUnitRange{<:Integer},N}}
) where {T,N}
  return SparseArrayDOK{T,N}(undef_blocks, ax)
end
function SparseArraysBase.SparseArrayDOK{T}(
  ::UndefBlocksInitializer, ax::Vararg{AbstractUnitRange{<:Integer},N}
) where {T,N}
  return SparseArrayDOK{T,N}(undef_blocks, ax)
end
function SparseArraysBase.SparseArrayDOK{T}(
  ::UndefBlocksInitializer,
  dims::Tuple{AbstractVector{<:Integer},Vararg{AbstractVector{<:Integer}}},
) where {T}
  return SparseArrayDOK{T}(undef_blocks, blockedrange.(dims))
end
function SparseArraysBase.SparseArrayDOK{T}(
  ::UndefBlocksInitializer,
  dim1::AbstractVector{<:Integer},
  dim_rest::AbstractVector{<:Integer}...,
) where {T}
  return SparseArrayDOK{T}(undef_blocks, (dim1, dim_rest...))
end

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
function sparsemortar(
  blocks::AbstractArray{<:AbstractArray{T,N},N},
  axes::Vararg{AbstractUnitRange{<:Integer},N},
) where {T,N}
  return sparsemortar(blocks, axes)
end
function sparsemortar(
  blocks::AbstractArray{<:AbstractArray{T,N},N},
  dims::Tuple{AbstractVector{<:Integer},Vararg{AbstractVector{<:Integer}}},
) where {T,N}
  return sparsemortar(blocks, blockedrange.(dims))
end
function sparsemortar(
  blocks::AbstractArray{<:AbstractArray{T,N},N},
  dim1::AbstractVector{<:Integer},
  dim_rest::AbstractVector{<:Integer}...,
) where {T,N}
  return sparsemortar(blocks, (dim1, dim_rest...))
end

@doc """
    BlockSparseArray{T}(undef, dims)
    BlockSparseArray{T,N}(undef, dims)
    BlockSparseArray{T,N,A}(undef, dims)

Construct an uninitialized N-dimensional BlockSparseArray containing elements of type T. `dims` should be a list
of block lengths in each dimension or a list of blocked ranges representing the axes.
""" BlockSparseArray

function BlockSparseArray{T,N,A}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange{<:Integer},N}}
) where {T,N,A<:AbstractArray{T,N}}
  return _BlockSparseArray(SparseArrayDOK{A}(undef_blocks, axes), axes)
end

function BlockSparseArray{T,N,A}(
  ::UndefInitializer, axes::Vararg{AbstractUnitRange{<:Integer},N}
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray{T,N,A}(undef, axes)
end

function BlockSparseArray{T,N,A}(
  ::UndefInitializer,
  dims::Tuple{AbstractVector{<:Integer},Vararg{AbstractVector{<:Integer}}},
) where {T,N,A<:AbstractArray{T,N}}
  length(dims) == N ||
    throw(ArgumentError("Length of dims doesn't match number of dimensions."))
  return BlockSparseArray{T,N,A}(undef, blockedrange.(dims))
end

function BlockSparseArray{T,N,A}(
  ::UndefInitializer,
  dim1::AbstractVector{<:Integer},
  dim_rest::AbstractVector{<:Integer}...,
) where {T,N,A<:AbstractArray{T,N}}
  return BlockSparseArray{T,N,A}(undef, (dim1, dim_rest...))
end

function similartype_unchecked(
  A::Type{<:AbstractArray{T}}, axt::Type{<:Tuple{Vararg{Any,N}}}
) where {T,N}
  A′ = Base.promote_op(similar, A, axt)
  return !isconcretetype(A′) ? Array{T,N} : A′
end

function BlockSparseArray{T,N}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange{<:Integer},N}}
) where {T,N}
  axt = Tuple{blockaxistype.(axes)...}
  # Ideally we would use:
  # ```julia
  # A = similartype(Array{T}, axt)
  # ```
  # but that doesn't work when `similar` isn't defined or
  # isn't type stable.
  A = similartype_unchecked(Array{T}, axt)
  return BlockSparseArray{T,N,A}(undef, axes)
end

function BlockSparseArray{T,N}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange{<:Integer}}}
) where {T,N}
  return throw(ArgumentError("Length of axes doesn't match number of dimensions."))
end

function BlockSparseArray{T,N}(
  ::UndefInitializer, axes::Vararg{AbstractUnitRange{<:Integer},N}
) where {T,N}
  return BlockSparseArray{T,N}(undef, axes)
end

function BlockSparseArray{T,N}(
  ::UndefInitializer,
  dims::Tuple{AbstractVector{<:Integer},Vararg{AbstractVector{<:Integer}}},
) where {T,N}
  return BlockSparseArray{T,N}(undef, blockedrange.(dims))
end

function BlockSparseArray{T,N}(
  ::UndefInitializer,
  dim1::AbstractVector{<:Integer},
  dim_rest::AbstractVector{<:Integer}...,
) where {T,N}
  return BlockSparseArray{T,N}(undef, (dim1, dim_rest...))
end

function BlockSparseArray{T}(
  ::UndefInitializer,
  dims::Tuple{AbstractVector{<:Integer},Vararg{AbstractVector{<:Integer}}},
) where {T}
  return BlockSparseArray{T,length(dims)}(undef, dims)
end

function BlockSparseArray{T}(
  ::UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange{<:Integer}}}
) where {T}
  return BlockSparseArray{T,length(axes)}(undef, axes)
end

function BlockSparseArray{T}(
  ::UndefInitializer,
  dim1::AbstractVector{<:Integer},
  dim_rest::AbstractVector{<:Integer}...,
) where {T}
  return BlockSparseArray{T}(undef, (dim1, dim_rest...))
end

function BlockSparseArray{T}(
  ::UndefInitializer, axes::Vararg{AbstractUnitRange{<:Integer}}
) where {T}
  return BlockSparseArray{T}(undef, axes)
end

# Convenient constructors.
function blocksparsezeros(elt::Type, axes...)
  return BlockSparseArray{elt}(undef, axes...)
end
function blocksparsezeros(::BlockType{A}, axes...) where {A<:AbstractArray}
  # TODO: Use:
  # ```julia
  # B = similartype(A, Type{eltype(A)}, Tuple{blockaxistype.(axes)...})
  # BlockSparseArray{eltype(A),length(axes),B}(undef, axes...)
  # ```
  # to make a bit more generic.
  return BlockSparseArray{eltype(A),ndims(A),A}(undef, axes...)
end
function blocksparse(d::Dict{<:Block,<:AbstractArray}, axes...)
  a = blocksparsezeros(BlockType(valtype(d)), axes...)
  for I in eachindex(d)
    a[I] = d[I]
  end
  return a
end

# Base `AbstractArray` interface
Base.axes(a::BlockSparseArray) = a.axes

# BlockArrays `AbstractBlockArray` interface.
# This is used by `blocks(::AnyAbstractBlockSparseArray)`.
@interface ::AbstractBlockSparseArrayInterface BlockArrays.blocks(a::BlockSparseArray) =
  a.blocks

function blocktype(
  arraytype::Type{<:BlockSparseArray{T,N,A}}
) where {T,N,A<:AbstractArray{T,N}}
  return A
end

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
