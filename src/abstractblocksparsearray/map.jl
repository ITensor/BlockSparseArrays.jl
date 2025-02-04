using ArrayLayouts: LayoutArray
using LinearAlgebra: Adjoint, Transpose

function Base.map!(f, a_dest::AbstractArray, a_srcs::AnyAbstractBlockSparseArray...)
  @interface interface(a_dest, a_srcs...) map!(f, a_dest, a_srcs...)
  return a_dest
end
function Base.map!(f, a_dest::AnyAbstractBlockSparseArray, a_srcs::AbstractArray...)
  @interface interface(a_dest, a_srcs...) map!(f, a_dest, a_srcs...)
  return a_dest
end
function Base.map!(
  f, a_dest::AnyAbstractBlockSparseArray, a_srcs::AnyAbstractBlockSparseArray...
)
  @interface interface(a_dest, a_srcs...) map!(f, a_dest, a_srcs...)
  return a_dest
end

function Base.map(f, as::Vararg{AnyAbstractBlockSparseArray})
  return f.(as...)
end

function Base.copy!(a_dest::AbstractArray, a_src::AnyAbstractBlockSparseArray)
  return @interface interface(a_src) copy!(a_dest, a_src)
end

function Base.copyto!(a_dest::AbstractArray, a_src::AnyAbstractBlockSparseArray)
  return @interface interface(a_src) copyto!(a_dest, a_src)
end

# Fix ambiguity error
function Base.copyto!(a_dest::LayoutArray, a_src::AnyAbstractBlockSparseArray)
  return @interface interface(a_src) copyto!(a_dest, a_src)
end

function Base.copyto!(
  a_dest::AbstractMatrix, a_src::Transpose{T,<:AbstractBlockSparseMatrix{T}}
) where {T}
  return @interface interface(a_src) copyto!(a_dest, a_src)
end

function Base.copyto!(
  a_dest::AbstractMatrix, a_src::Adjoint{T,<:AbstractBlockSparseMatrix{T}}
) where {T}
  return @interface interface(a_src) copyto!(a_dest, a_src)
end

# This avoids going through the generic version that calls `Base.permutedims!`,
# which eventually calls block sparse `map!`, which involves slicing operations
# that are not friendly to GPU (since they involve `SubArray` wrapping
# `PermutedDimsArray`).
# TODO: Handle slicing better in `map!` so that this can be removed.
function Base.permutedims(a::AnyAbstractBlockSparseArray, perm)
  @interface interface(a) permutedims(a, perm)
end

# The `::AbstractBlockSparseArrayInterface` version
# has a special case for when `a_dest` and `PermutedDimsArray(a_src, perm)`
# have the same blocking, and therefore can just use:
# ```julia
# permutedims!(blocks(a_dest), blocks(a_src), perm)
# ```
# TODO: Handle slicing better in `map!` so that this can be removed.
function Base.permutedims!(a_dest, a_src::AnyAbstractBlockSparseArray, perm)
  return @interface interface(a_src) permutedims!(a_dest, a_src, perm)
end

function Base.mapreduce(f, op, as::AnyAbstractBlockSparseArray...; kwargs...)
  return @interface interface(as...) mapreduce(f, op, as...; kwargs...)
end

function Base.iszero(a::AnyAbstractBlockSparseArray)
  return @interface interface(a) iszero(a)
end

function Base.isreal(a::AnyAbstractBlockSparseArray)
  return @interface interface(a) isreal(a)
end
