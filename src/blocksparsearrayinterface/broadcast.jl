using Base.Broadcast: AbstractArrayStyle, BroadcastStyle, Broadcasted
using GPUArraysCore: @allowscalar
using MapBroadcast: Mapped
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
function BlockSparseArrayStyle{N}() where {N}
    return BlockSparseArrayStyle{N}(Base.Broadcast.DefaultArrayStyle{N}())
end
BlockSparseArrayStyle(::Val{N}) where {N} = BlockSparseArrayStyle{N}()
BlockSparseArrayStyle{M}(::Val{N}) where {M, N} = BlockSparseArrayStyle{N}()
function BlockSparseArrayStyle{M, B}(::Val{N}) where {M, B <: AbstractArrayStyle{M}, N}
    return BlockSparseArrayStyle{N}(B(Val(N)))
end

function blockstyle(
        ::AbstractBlockSparseArrayStyle{N, B}
    ) where {N, B <: Base.Broadcast.AbstractArrayStyle{N}}
    return B()
end

function Base.Broadcast.BroadcastStyle(
        style1::AbstractBlockSparseArrayStyle,
        style2::AbstractBlockSparseArrayStyle
    )
    style = Base.Broadcast.result_style(blockstyle(style1), blockstyle(style2))
    return BlockSparseArrayStyle(style)
end

function Base.Broadcast.BroadcastStyle(
        a::BlockSparseArrayStyle,
        ::Base.Broadcast.DefaultArrayStyle{0}
    )
    return a
end
function Base.Broadcast.BroadcastStyle(
        ::BlockSparseArrayStyle{N}, a::Base.Broadcast.DefaultArrayStyle
    ) where {N}
    return Base.Broadcast.BroadcastStyle(Base.Broadcast.DefaultArrayStyle{N}(), a)
end
function Base.Broadcast.BroadcastStyle(
        ::BlockSparseArrayStyle{N}, ::Base.Broadcast.Style{Tuple}
    ) where {N}
    return Base.Broadcast.DefaultArrayStyle{N}()
end

function Base.similar(bc::Broadcasted{<:BlockSparseArrayStyle}, elt::Type, ax)
    # Find the first array in the broadcast expression.
    # TODO: Make this more generic, base it off sure this handles GPU arrays properly.
    bc′ = Base.Broadcast.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractArray, bc′.args)]
    return similar(arg, elt, ax)
end

# Catches cases like `dest .= value` or `dest .= value1 .+ value2`.
# If the RHS is zero, this makes sure that the storage is emptied,
# which is logic that is handled by `fill!`.
const copyto!_blocksparse = blocksparse_style(copyto!)
const fill!_blocksparse = blocksparse_style(fill!)
function copyto!_blocksparse(
        dest::AbstractArray,
        bc::Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}
    )
    # `[]` is used to unwrap zero-dimensional arrays.
    bcf = Base.Broadcast.flatten(bc)
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
        dest::AbstractArray{<:Any, N}, bc::Broadcasted{BlockSparseArrayStyle{N}}
    ) where {N}
    return copyto!_blocksparse(dest, bc)
end
