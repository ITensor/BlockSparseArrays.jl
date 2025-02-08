using BlockArrays:
  BlockArrays, AbstractBlockArray, Block, BlockIndex, BlockedUnitRange, blocks

abstract type AbstractBlockSparseArray{T,N} <: AbstractBlockArray{T,N} end

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
  return @interface BlockSparseArrayInterface() getindex(a, I...)
end

# Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.getindex(a::AbstractBlockSparseArray{<:Any,0})
  return @interface BlockSparseArrayInterface() getindex(a)
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
##   ## return @interface BlockSparseArrayInterface() getindex(a, I...)
##   return ArrayLayouts.layout_getindex(a, I...)
## end

# Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.setindex!(
  a::AbstractBlockSparseArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  @interface BlockSparseArrayInterface() setindex!(a, value, I...)
  return a
end

# Fix ambiguity error.
function Base.setindex!(a::AbstractBlockSparseArray{<:Any,0}, value)
  @interface BlockSparseArrayInterface() setindex!(a, value)
  return a
end

# Catch zero-dimensional case to avoid scalar indexing.
function Base.setindex!(a::AbstractBlockSparseArray{<:Any,0}, value, ::Block{0})
  blocks(a)[] = value
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

using TypeParameterAccessors: unspecify_type_parameters
function show_typeof_blocksparse(io::IO, a::AbstractBlockSparseArray)
  Base.show(io, unspecify_type_parameters(typeof(a)))
  print(io, '{')
  show(io, eltype(a))
  print(io, ", ")
  show(io, ndims(a))
  print(io, ", ")
  show(io, blocktype(a))
  print(io, ", …")
  print(io, '}')
  return nothing
end

# Copied from `BlockArrays.jl`.
block2string(b, s) = string(join(map(string, b), '×'), "-blocked ", Base.dims2string(s))

function summary_blocksparse(io::IO, a::AbstractArray)
  print(io, block2string(blocksize(a), size(a)))
  print(io, ' ')
  show_typeof_blocksparse(io, a)
  return nothing
end

function Base.summary(io::IO, a::AbstractBlockSparseArray)
  summary_blocksparse(io, a)
  return nothing
end

function Base.showarg(io::IO, a::AbstractBlockSparseArray, toplevel::Bool)
  if toplevel
    show_typeof_blocksparse(io, a)
  else
    print(io, "::")
    show_typeof_blocksparse(io, a)
  end
  return nothing
end
