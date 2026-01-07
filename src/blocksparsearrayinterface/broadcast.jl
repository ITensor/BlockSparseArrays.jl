using Base.Broadcast: BroadcastStyle, Broadcasted
using GPUArraysCore: @allowscalar
using MapBroadcast: Mapped

module Broadcast
    using Base.Broadcast: AbstractArrayStyle
    abstract type AbstractBlockSparseArrayStyle{N, B <: AbstractArrayStyle{N}} <:
    AbstractArrayStyle{N} end
    struct BlockSparseArrayStyle{N, B <: AbstractArrayStyle{N}} <:
        AbstractBlockSparseArrayStyle{N, B}
        blockstyle::B
    end
    function BlockSparseArrayStyle{N}(blockstyle::AbstractArrayStyle{N}) where {N}
        return BlockSparseArrayStyle{N, typeof(blockstyle)}(blockstyle)
    end
    function BlockSparseArrayStyle{N, B}() where {N, B <: AbstractArrayStyle{N}}
        return BlockSparseArrayStyle{N, B}(B())
    end
    BlockSparseArrayStyle{N}() where {N} = BlockSparseArrayStyle{N}(DefaultArrayStyle{N}())
    BlockSparseArrayStyle(::Val{N}) where {N} = BlockSparseArrayStyle{N}()
    BlockSparseArrayStyle{M}(::Val{N}) where {M, N} = BlockSparseArrayStyle{N}()
    function BlockSparseArrayStyle{M, B}(::Val{N}) where {M, B <: AbstractArrayStyle{M}, N}
        return BlockSparseArrayStyle{N}(B(Val(N)))
    end
end

function blockstyle(
        ::Broadcast.AbstractBlockSparseArrayStyle{N, B},
    ) where {N, B <: Base.Broadcast.AbstractArrayStyle{N}}
    return B()
end

function Base.Broadcast.BroadcastStyle(
        style1::Broadcast.AbstractBlockSparseArrayStyle,
        style2::Broadcast.AbstractBlockSparseArrayStyle,
    )
    style = Base.Broadcast.result_style(blockstyle(style1), blockstyle(style2))
    return Broadcast.BlockSparseArrayStyle(style)
end

Base.Broadcast.BroadcastStyle(a::Broadcast.BlockSparseArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
function Base.Broadcast.BroadcastStyle(
        ::Broadcast.BlockSparseArrayStyle{N}, a::Base.Broadcast.DefaultArrayStyle
    ) where {N}
    return Base.Broadcast.BroadcastStyle(Base.Broadcast.DefaultArrayStyle{N}(), a)
end
function Base.Broadcast.BroadcastStyle(
        ::Broadcast.BlockSparseArrayStyle{N}, ::Base.Broadcast.Style{Tuple}
    ) where {N}
    return Base.Broadcast.DefaultArrayStyle{N}()
end

function Base.similar(bc::Broadcasted{<:Broadcast.BlockSparseArrayStyle}, elt::Type, ax)
    # TODO: Make this more generic, base it off sure this handles GPU arrays properly.
    m = Mapped(bc)
    return similar(first(m.args), elt, ax)
end

# Catches cases like `dest .= value` or `dest .= value1 .+ value2`.
# If the RHS is zero, this makes sure that the storage is emptied,
# which is logic that is handled by `fill!`.
const copyto!_blocksparse = blocksparse_style(copyto!)
const fill!_blocksparse = blocksparse_style(fill!)
function copyto!_blocksparse(dest::AbstractArray, bc::Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}})
    # `[]` is used to unwrap zero-dimensional arrays.
    bcf = Broadcast.flatten(bc)
    value = @allowscalar bcf.f(map(arg -> arg[], bcf.args)...)
    return fill!_blocksparse(dest, value)
end

function copyto!_blocksparse(
        dest::AbstractArray, bc::Broadcasted
    )
    # convert to map
    # flatten and only keep the AbstractArray arguments
    m = Mapped(bc)
    map!(m.f, dest, m.args...)
    return dest
end

# Broadcasting implementation
function Base.copyto!(
        dest::AbstractArray{<:Any, N}, bc::Broadcasted{Broadcast.BlockSparseArrayStyle{N}}
    ) where {N}
    return copyto!_blocksparse(dest, bc)
end
