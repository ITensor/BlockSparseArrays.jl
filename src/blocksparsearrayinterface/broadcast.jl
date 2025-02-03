using Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted
using MapBroadcast: Mapped
using DerivableInterfaces: DerivableInterfaces, @interface

abstract type AbstractBlockSparseArrayStyle{N} <: AbstractArrayStyle{N} end

function DerivableInterfaces.interface(::Type{<:AbstractBlockSparseArrayStyle})
  return BlockSparseArrayInterface()
end

struct BlockSparseArrayStyle{N} <: AbstractBlockSparseArrayStyle{N} end

# Define for new sparse array types.
# function Broadcast.BroadcastStyle(arraytype::Type{<:MyBlockSparseArray})
#   return BlockSparseArrayStyle{ndims(arraytype)}()
# end

BlockSparseArrayStyle(::Val{N}) where {N} = BlockSparseArrayStyle{N}()
BlockSparseArrayStyle{M}(::Val{N}) where {M,N} = BlockSparseArrayStyle{N}()

Broadcast.BroadcastStyle(a::BlockSparseArrayStyle, ::DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(
  ::BlockSparseArrayStyle{N}, a::DefaultArrayStyle
) where {N}
  return BroadcastStyle(DefaultArrayStyle{N}(), a)
end
function Broadcast.BroadcastStyle(
  ::BlockSparseArrayStyle{N}, ::Broadcast.Style{Tuple}
) where {N}
  return DefaultArrayStyle{N}()
end

function Base.similar(bc::Broadcasted{<:BlockSparseArrayStyle}, elt::Type)
  # TODO: Make sure this handles GPU arrays properly.
  m = Mapped(bc)
  return similar(first(m.args), elt, combine_axes(axes.(m.args)...))
end

# Broadcasting implementation
# TODO: Delete this in favor of `DerivableInterfaces` version.
function Base.copyto!(
  dest::AbstractArray{<:Any,N}, bc::Broadcasted{BlockSparseArrayStyle{N}}
) where {N}
  # convert to map
  # flatten and only keep the AbstractArray arguments
  m = Mapped(bc)
  @interface interface(dest, bc) map!(m.f, dest, m.args...)
  return dest
end

# Broadcasting implementation
# TODO: Delete this in favor of `DerivableInterfaces` version.
function Base.copyto!(dest::AnyAbstractBlockSparseArray, bc::Broadcasted)
  # convert to map
  # flatten and only keep the AbstractArray arguments
  m = Mapped(bc)
  # TODO: Include `bc` when determining interface, currently
  # `interface(::Type{<:Base.Broadcast.DefaultArrayStyle})`
  # isn't defined.
  @interface interface(dest) map!(m.f, dest, m.args...)
  return dest
end

# Broadcasting implementation
# TODO: Delete this in favor of `DerivableInterfaces` version.
function Base.copyto!(
  dest::AnyAbstractBlockSparseArray, bc::Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}
)
  # convert to map
  # flatten and only keep the AbstractArray arguments
  m = Mapped(bc)
  # TODO: Include `bc` when determining interface, currently
  # `interface(::Type{<:Base.Broadcast.DefaultArrayStyle})`
  # isn't defined.
  @interface interface(dest) map!(m.f, dest, m.args...)
  return dest
end

# Broadcasting implementation
# TODO: Delete this in favor of `DerivableInterfaces` version.
function Base.copyto!(
  dest::AnyAbstractBlockSparseArray{<:Any,N}, bc::Broadcasted{BlockSparseArrayStyle{N}}
) where {N}
  # convert to map
  # flatten and only keep the AbstractArray arguments
  m = Mapped(bc)
  @interface interface(dest, bc) map!(m.f, dest, m.args...)
  return dest
end
