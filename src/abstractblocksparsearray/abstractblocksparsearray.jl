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
  return @interface interface(a) getindex(a, I...)
end

# Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.getindex(a::AbstractBlockSparseArray{<:Any,0})
  return @interface interface(a) getindex(a)
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
##   ## return @interface interface(a) getindex(a, I...)
##   return ArrayLayouts.layout_getindex(a, I...)
## end

# Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.setindex!(
  a::AbstractBlockSparseArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  @interface interface(a) setindex!(a, value, I...)
  return a
end

# Fix ambiguity error.
function Base.setindex!(a::AbstractBlockSparseArray{<:Any,0}, value)
  @interface interface(a) setindex!(a, value)
  return a
end

# Catch zero-dimensional case to avoid scalar indexing.
function Base.setindex!(a::AbstractBlockSparseArray{<:Any,0}, value, ::Block{0})
  blocks(a)[] = value
  return a
end

# Custom `_convert` works around the issue that
# `convert(::Type{<:Diagonal}, ::AbstractMatrix)` isnt' defined
# in Julia v1.10 (https://github.com/JuliaLang/julia/pull/48895,
# https://github.com/JuliaLang/julia/pull/52487).
# TODO: Delete once we drop support for Julia v1.10.
_convert(::Type{T}, a::AbstractArray) where {T} = convert(T, a)
using LinearAlgebra: LinearAlgebra, Diagonal, diag, isdiag
_construct(T::Type{<:Diagonal}, a::AbstractMatrix) = T(diag(a))
function _convert(T::Type{<:Diagonal}, a::AbstractMatrix)
  LinearAlgebra.checksquare(a)
  return isdiag(a) ? _construct(T, a) : throw(InexactError(:convert, T, a))
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
  if isstored(a, I...)
    # This writes into existing blocks, or constructs blocks
    # using the axes.
    aI = @view! a[I...]
    aI .= value
  else
    # Custom `_convert` works around the issue that
    # `convert(::Type{<:Diagonal}, ::AbstractMatrix)` isnt' defined
    # in Julia v1.10 (https://github.com/JuliaLang/julia/pull/48895,
    # https://github.com/JuliaLang/julia/pull/52487).
    # TODO: Delete `_convert` once we drop support for Julia v1.10.
    blocks(a)[Int.(I)...] = _convert(blocktype(a), value)
  end
  return a
end

# Copy of `Base.dims2string` defined in `show.jl`.
function dims_to_string(d)
  isempty(d) && return "0-dimensional"
  length(d) == 1 && return "$(d[1])-element"
  return join(map(string, d), '×')
end

# Copy of `BlockArrays.block2string` from `BlockArrays.jl`.
block_to_string(b, s) = string(join(map(string, b), '×'), "-blocked ", dims_to_string(s))

using TypeParameterAccessors: type_parameters, unspecify_type_parameters
function concretetype_to_string_truncated(type::Type; param_truncation_length=typemax(Int))
  isconcretetype(type) || throw(ArgumentError("Type must be concrete."))
  alias = Base.make_typealias(type)
  base_type, params = if isnothing(alias)
    unspecify_type_parameters(type), type_parameters(type)
  else
    base_type_globalref, params_svec = alias
    base_type_globalref.name, params_svec
  end
  str = string(base_type)
  if isempty(params)
    return str
  end
  str *= '{'
  param_strings = map(params) do param
    param_string = string(param)
    if length(param_string) > param_truncation_length
      return "…"
    end
    return param_string
  end
  str *= join(param_strings, ", ")
  str *= '}'
  return str
end

function Base.summary(io::IO, a::AbstractBlockSparseArray)
  print(io, block_to_string(blocksize(a), size(a)))
  print(io, ' ')
  print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length=40))
  return nothing
end

function Base.showarg(io::IO, a::AbstractBlockSparseArray, toplevel::Bool)
  !toplevel && print(io, "::")
  print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length=40))
  return nothing
end
