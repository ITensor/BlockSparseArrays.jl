module BlockSparseArraysGradedUnitRangesExt

using BlockSparseArrays: BlockSparseArray
using GradedUnitRanges: AbstractGradedUnitRange

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
  return BlockSparseArray{elt,length(axes),typeof(a)}(axes)
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
  axes::Tuple{
    AbstractGradedUnitRange,AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}
  },
)
  return similar_blocksparse(a, elt, axes)
end

function Base.getindex(a::AbstractArray, I::AbstractGradedUnitRange...)
  a′ = similar(a, only.(axes.(I))...)
  a′ .= a
  return a′
end

end
