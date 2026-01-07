using BlockArrays: AbstractBlockedUnitRange, BlockSlice
using Base.Broadcast: BroadcastStyle

function Base.Broadcast.BroadcastStyle(arraytype::Type{<:AnyAbstractBlockSparseArray})
    return Broadcast.BlockSparseArrayStyle(BroadcastStyle(blocktype(arraytype)))
end

# Fix ambiguity error with `BlockArrays`.
function Base.Broadcast.BroadcastStyle(
        arraytype::Type{
            <:SubArray{
                <:Any,
                <:Any,
                <:AbstractBlockSparseArray,
                <:Tuple{BlockSlice{<:Any, <:Any, <:AbstractBlockedUnitRange}, Vararg{Any}},
            },
        },
    )
    return BlockSparseArrayStyle{ndims(arraytype)}()
end
function Base.Broadcast.BroadcastStyle(
        arraytype::Type{
            <:SubArray{
                <:Any,
                <:Any,
                <:AbstractBlockSparseArray,
                <:Tuple{
                    BlockSlice{<:Any, <:Any, <:AbstractBlockedUnitRange},
                    BlockSlice{<:Any, <:Any, <:AbstractBlockedUnitRange},
                    Vararg{Any},
                },
            },
        },
    )
    return BlockSparseArrayStyle{ndims(arraytype)}()
end
function Base.Broadcast.BroadcastStyle(
        arraytype::Type{
            <:SubArray{
                <:Any,
                <:Any,
                <:AbstractBlockSparseArray,
                <:Tuple{Any, BlockSlice{<:Any, <:Any, <:AbstractBlockedUnitRange}, Vararg{Any}},
            },
        },
    )
    return BlockSparseArrayStyle{ndims(arraytype)}()
end

# These catch cases that aren't caught by the standard
# `BlockSparseArrayStyle` definition, and also fix
# ambiguity issues.
function Base.copyto!(dest::AnyAbstractBlockSparseArray, bc::Broadcasted)
    return copyto!_blocksparse(dest, bc)
end
function Base.copyto!(
        dest::AnyAbstractBlockSparseArray, bc::Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}
    )
    return copyto!_blocksparse(dest, bc)
end
function Base.copyto!(
        dest::AnyAbstractBlockSparseArray{<:Any, N}, bc::Broadcasted{<:Broadcast.BlockSparseArrayStyle{N}}
    ) where {N}
    return copyto!_blocksparse(dest, bc)
end
