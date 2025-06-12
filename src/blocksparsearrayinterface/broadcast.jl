using Base.Broadcast:
  Broadcast, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted
using GPUArraysCore: @allowscalar
using MapBroadcast: Mapped
using DerivableInterfaces: DerivableInterfaces, @interface

abstract type AbstractBlockSparseArrayStyle{N,B<:AbstractArrayStyle{N}} <:
              AbstractArrayStyle{N} end

blockstyle(::AbstractBlockSparseArrayStyle{N,B}) where {N,B<:AbstractArrayStyle{N}} = B()

function Broadcast.BroadcastStyle(
  style1::AbstractBlockSparseArrayStyle, style2::AbstractBlockSparseArrayStyle
)
  style = Broadcast.result_style(blockstyle(style1), blockstyle(style2))
  return BlockSparseArrayStyle(style)
end

function DerivableInterfaces.interface(
  ::Type{<:AbstractBlockSparseArrayStyle{N,B}}
) where {N,B<:AbstractArrayStyle{N}}
  return BlockSparseArrayInterface(interface(B))
end

struct BlockSparseArrayStyle{N,B<:AbstractArrayStyle{N}} <:
       AbstractBlockSparseArrayStyle{N,B}
  blockstyle::B
end
function BlockSparseArrayStyle{N}(blockstyle::AbstractArrayStyle{N}) where {N}
  return BlockSparseArrayStyle{N,typeof(blockstyle)}(blockstyle)
end

function BlockSparseArrayStyle{N,B}() where {N,B<:AbstractArrayStyle{N}}
  return BlockSparseArrayStyle{N,B}(B())
end
BlockSparseArrayStyle{N}() where {N} = BlockSparseArrayStyle{N}(DefaultArrayStyle{N}())
BlockSparseArrayStyle(::Val{N}) where {N} = BlockSparseArrayStyle{N}()
BlockSparseArrayStyle{M}(::Val{N}) where {M,N} = BlockSparseArrayStyle{N}()
function BlockSparseArrayStyle{M,B}(::Val{N}) where {M,B<:AbstractArrayStyle{M},N}
  return BlockSparseArrayStyle{N}(B(Val(N)))
end

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

function Base.similar(bc::Broadcasted{<:BlockSparseArrayStyle}, elt::Type, ax)
  # TODO: Make this more generic, base it off sure this handles GPU arrays properly.
  m = Mapped(bc)
  return similar(first(m.args), elt, ax)
end

# Catches cases like `dest .= value` or `dest .= value1 .+ value2`.
# If the RHS is zero, this makes sure that the storage is emptied,
# which is logic that is handled by `fill!`.
function copyto_blocksparse!(dest::AbstractArray, bc::Broadcasted{<:AbstractArrayStyle{0}})
  # `[]` is used to unwrap zero-dimensional arrays.
  value = @allowscalar bc.f(bc.args...)[]
  return @interface BlockSparseArrayInterface() fill!(dest, value)
end

# Broadcasting implementation
# TODO: Delete this in favor of `DerivableInterfaces` version.
function copyto_blocksparse!(dest::AbstractArray, bc::Broadcasted)
  # convert to map
  # flatten and only keep the AbstractArray arguments
  m = Mapped(bc)
  @interface interface(dest, bc) map!(m.f, dest, m.args...)
  return dest
end

function Base.copyto!(
  dest::AbstractArray{<:Any,N}, bc::Broadcasted{BlockSparseArrayStyle{N}}
) where {N}
  copyto_blocksparse!(dest, bc)
  return dest
end
