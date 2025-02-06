module BlockSparseArraysGradedUnitRangesExt

using BlockSparseArrays: AnyAbstractBlockSparseArray, BlockSparseArray, blocktype
using GradedUnitRanges: AbstractGradedUnitRange
using TypeParameterAccessors: set_ndims, unwrap_array_type

# A block spare array similar to the input (dense) array.
# TODO: Make `BlockSparseArrays.blocksparse_similar` more general and use that,
# and also turn it into an DerivableInterfaces.jl-based interface function.
function similar_blocksparse(
  a::AbstractArray,
  elt::Type,
  axes::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}},
)
  # TODO: Probably need to unwrap the type of `a` in certain cases
  # to make a proper block type.
  return BlockSparseArray{
    elt,length(axes),set_ndims(unwrap_array_type(blocktype(a)), length(axes))
  }(
    axes
  )
end

function Base.similar(
  a::AbstractArray,
  elt::Type,
  axes::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}},
)
  return similar_blocksparse(a, elt, axes)
end

# Fix ambiguity error with `BlockArrays.jl`.
function Base.similar(
  a::StridedArray,
  elt::Type,
  axes::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}},
)
  return similar_blocksparse(a, elt, axes)
end

# Fix ambiguity error with `BlockArrays.jl`.
function Base.similar(
  a::StridedArray,
  elt::Type,
  axes::Tuple{
    AbstractGradedUnitRange,AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}
  },
)
  return similar_blocksparse(a, elt, axes)
end

# Fix ambiguity error with `BlockSparseArrays.jl`.
function Base.similar(
  a::AnyAbstractBlockSparseArray,
  elt::Type,
  axes::Tuple{
    AbstractGradedUnitRange,AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}
  },
)
  return similar_blocksparse(a, elt, axes)
end

function getindex_blocksparse(a::AbstractArray, I::AbstractUnitRange...)
  a′ = similar(a, only.(axes.(I))...)
  a′ .= a
  return a′
end

function Base.getindex(
  a::AbstractArray, I1::AbstractGradedUnitRange, I_rest::AbstractGradedUnitRange...
)
  return getindex_blocksparse(a, I1, I_rest...)
end

# Fix ambiguity error with Base.
function Base.getindex(a::Vector, I::AbstractGradedUnitRange)
  return getindex_blocksparse(a, I)
end

# Fix ambiguity error with BlockSparseArrays.jl.
function Base.getindex(
  a::AnyAbstractBlockSparseArray,
  I1::AbstractGradedUnitRange,
  I_rest::AbstractGradedUnitRange...,
)
  return getindex_blocksparse(a, I1, I_rest...)
end

# Fix ambiguity error with BlockSparseArrays.jl.
function Base.getindex(
  a::AnyAbstractBlockSparseArray{<:Any,2},
  I1::AbstractGradedUnitRange,
  I2::AbstractGradedUnitRange,
)
  return getindex_blocksparse(a, I1, I2)
end

end
