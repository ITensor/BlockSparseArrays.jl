using ArrayLayouts: ArrayLayouts, zero!
using BlockArrays:
  BlockArrays,
  AbstractBlockVector,
  Block,
  BlockIndex,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedUnitRange,
  BlockedVector,
  block,
  blockcheckbounds,
  blockisequal,
  blocklengths,
  blocklength,
  blocks,
  findblockindex
using DerivableInterfaces: DerivableInterfaces, @interface, DefaultArrayInterface
using LinearAlgebra: Adjoint, Transpose
using SparseArraysBase:
  AbstractSparseArrayInterface,
  getstoredindex,
  getunstoredindex,
  eachstoredindex,
  perm,
  iperm,
  storedlength,
  storedvalues

# Like `SparseArraysBase.eachstoredindex` but
# at the block level, i.e. iterates over the
# stored block locations.
function eachblockstoredindex(a::AbstractArray)
  # TODO: Use `Iterators.map`.
  return Block.(Tuple.(eachstoredindex(blocks(a))))
end

# Like `BlockArrays.eachblock` but only iterating
# over stored blocks.
function eachstoredblock(a::AbstractArray)
  return storedvalues(blocks(a))
end

function blockstype(a::AbstractArray)
  return typeof(blocks(a))
end

#=
Ideally this would just be defined as `eltype(blockstype(a))`.
However, BlockArrays.jl doesn't make `eltype(blocks(a))` concrete
even when it could be
(https://github.com/JuliaArrays/BlockArrays.jl/blob/v1.4.0/src/blocks.jl#L71-L74):
```julia
julia> eltype(blocks(BlockArray(randn(2, 2), [1, 1], [1, 1])))
Matrix{Float64} (alias for Array{Float64, 2})

julia> eltype(blocks(BlockedArray(randn(2, 2), [1, 1], [1, 1])))
AbstractMatrix{Float64} (alias for AbstractArray{Float64, 2})

julia> eltype(blocks(randn(2, 2)))
AbstractMatrix{Float64} (alias for AbstractArray{Float64, 2})
```
Also note the current definition errors in the limit
when `blocks(a)` is empty, but even empty arrays generally
have at least one block:
```julia
julia> length(blocks(randn(0)))
1

julia> length(blocks(BlockVector{Float64}(randn(0))))
1

julia> length(blocks(BlockedVector{Float64}(randn(0))))
1
```
=#
function blocktype(a::AbstractArray)
  if isempty(blocks(a))
    error("`blocktype` can't be determined if `isempty(blocks(a))`.")
  end
  return mapreduce(typeof, promote_type, blocks(a))
end

using BlockArrays: BlockArray
blockstype(::Type{<:BlockArray{<:Any,<:Any,B}}) where {B} = B
blockstype(a::BlockArray) = blockstype(typeof(a))
blocktype(arraytype::Type{<:BlockArray}) = eltype(blockstype(arraytype))
blocktype(a::BlockArray) = eltype(blocks(a))

abstract type AbstractBlockSparseArrayInterface <: AbstractSparseArrayInterface end

# TODO: Also support specifying the `blocktype` along with the `eltype`.
function DerivableInterfaces.arraytype(::AbstractBlockSparseArrayInterface, T::Type)
  return BlockSparseArray{T}
end

struct BlockSparseArrayInterface <: AbstractBlockSparseArrayInterface end

@interface ::AbstractBlockSparseArrayInterface BlockArrays.blocks(a::AbstractArray) =
  error("Not implemented")

@interface ::AbstractBlockSparseArrayInterface function SparseArraysBase.isstored(
  a::AbstractArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  bI = BlockIndex(findblockindex.(axes(a), I))
  return isstored(blocks(a), bI.I...) && isstored(blocks(a)[bI.I...], bI.α...)
end

@interface ::AbstractBlockSparseArrayInterface function Base.getindex(
  a::AbstractArray{<:Any,N}, I::Vararg{Int,N}
) where {N}
  @boundscheck checkbounds(a, I...)
  return a[findblockindex.(axes(a), I)...]
end

# Fix ambiguity error.
@interface ::AbstractBlockSparseArrayInterface function Base.getindex(
  a::AbstractArray{<:Any,0}
)
  return a[Block()[]]
end

# a[1:2, 1:2]
# TODO: This definition means that the result of slicing a block sparse array
# with a non-blocked unit range is blocked. We may want to change that behavior,
# and make that explicit with `@blocked a[1:2, 1:2]`. See the discussion in
# https://github.com/JuliaArrays/BlockArrays.jl/issues/347 and also
# https://github.com/ITensor/ITensors.jl/issues/1336.
@interface ::AbstractBlockSparseArrayInterface function Base.to_indices(
  a, inds, I::Tuple{UnitRange{<:Integer},Vararg{Any}}
)
  bs1 = to_blockindices(inds[1], I[1])
  I1 = BlockSlice(bs1, blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# Special case when there is no blocking.
@interface ::AbstractBlockSparseArrayInterface function Base.to_indices(
  a,
  inds::Tuple{Base.OneTo{<:Integer},Vararg{Any}},
  I::Tuple{UnitRange{<:Integer},Vararg{Any}},
)
  return (inds[1][I[1]], to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# a[[Block(2), Block(1)], [Block(2), Block(1)]]
@interface ::AbstractBlockSparseArrayInterface function Base.to_indices(
  a, inds, I::Tuple{Vector{<:Block{1}},Vararg{Any}}
)
  I1 = BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# a[mortar([Block(1)[1:2], Block(2)[1:3]]), mortar([Block(1)[1:2], Block(2)[1:3]])]
# a[[Block(1)[1:2], Block(2)[1:3]], [Block(1)[1:2], Block(2)[1:3]]]
@interface ::AbstractBlockSparseArrayInterface function Base.to_indices(
  a, inds, I::Tuple{BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}},Vararg{Any}}
)
  I1 = BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# a[BlockVector([Block(2), Block(1)], [2]), BlockVector([Block(2), Block(1)], [2])]
# Permute and merge blocks.
# TODO: This isn't merging blocks yet, that needs to be implemented that.
@interface ::AbstractBlockSparseArrayInterface function Base.to_indices(
  a, inds, I::Tuple{AbstractBlockVector{<:Block{1}},Vararg{Any}}
)
  I1 = BlockIndices(I[1], blockedunitrange_getindices(inds[1], I[1]))
  return (I1, to_indices(a, Base.tail(inds), Base.tail(I))...)
end

# TODO: Need to implement this!
function block_merge end

@interface ::AbstractBlockSparseArrayInterface function Base.setindex!(
  a::AbstractArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  @boundscheck checkbounds(a, I...)
  a[findblockindex.(axes(a), I)...] = value
  return a
end

# Fix ambiguity error.
@interface ::AbstractBlockSparseArrayInterface function Base.setindex!(
  a::AbstractArray{<:Any,0}, value
)
  # TODO: Use `Block()[]` once https://github.com/JuliaArrays/BlockArrays.jl/issues/430
  # is fixed.
  a[BlockIndex()] = value
  return a
end

@interface ::AbstractBlockSparseArrayInterface function Base.setindex!(
  a::AbstractArray{<:Any,N}, value, I::BlockIndex{N}
) where {N}
  i = Int.(Tuple(block(I)))
  a_b = blocks(a)[i...]
  a_b[I.α...] = value
  # Set the block, required if it is structurally zero.
  blocks(a)[i...] = a_b
  return a
end

# Fix ambiguity error.
@interface ::AbstractBlockSparseArrayInterface function Base.setindex!(
  a::AbstractArray{<:Any,0}, value, I::BlockIndex{0}
)
  a_b = blocks(a)[]
  # `value[]` handles scalars and 0-dimensional arrays.
  a_b[] = value[]
  # Set the block, required if it is structurally zero.
  blocks(a)[] = a_b
  return a
end

# Version of `permutedims!` that assumes the destination and source
# have the same blocking.
# TODO: Delete this and handle this logic in block sparse `map[!]`, define
# `blockisequal_map[!]`.
# TODO: Maybe define a `BlockIsEqualInterface` for these kinds of functions.
function blockisequal_permutedims!(a_dest::AbstractArray, a_src::AbstractArray, perm)
  blocks(a_dest) .= blocks(PermutedDimsArray(a_src, perm))
  return a_dest
end

# We overload `permutedims` here so that we can assume the destination and source
# have the same blocking and avoid non-GPU friendly slicing operations in block sparse `map!`.
# TODO: Delete this and handle this logic in block sparse `map!`.
@interface ::AbstractBlockSparseArrayInterface function Base.permutedims(
  a::AbstractArray, perm
)
  a_dest = similar(PermutedDimsArray(a, perm))
  # TODO: Maybe define this as `@interface BlockIsEqualInterface() permutedims!(...)`.
  blockisequal_permutedims!(a_dest, a, perm)
  return a_dest
end

# We overload `permutedims!` here so that we can special case when the destination and source
# have the same blocking and avoid non-GPU friendly slicing operations in block sparse `map!`.
# TODO: Delete this and handle this logic in block sparse `map!`.
@interface ::AbstractBlockSparseArrayInterface function Base.permutedims!(
  a_dest::AbstractArray, a_src::AbstractArray, perm
)
  if all(blockisequal.(axes(a_dest), axes(PermutedDimsArray(a_src, perm))))
    # TODO: Maybe define this as `@interface BlockIsEqualInterface() permutedims!(...)`.
    blockisequal_permutedims!(a_dest, a_src, perm)
    return a_dest
  end
  @interface DefaultArrayInterface() permutedims!(a_dest, a_src, perm)
  return a_dest
end

@interface ::AbstractBlockSparseArrayInterface function Base.fill!(a::AbstractArray, value)
  # TODO: Only do this check if `value isa Number`?
  if iszero(value)
    zero!(a)
    return a
  end
  # TODO: Maybe use `map` over `blocks(a)` or something
  # like that.
  for b in BlockRange(a)
    a[b] .= value
  end
  return a
end

@interface ::AbstractBlockSparseArrayInterface function ArrayLayouts.zero!(a::AbstractArray)
  # This will try to empty the storage if possible.
  zero!(blocks(a))
  return a
end

# TODO: Rename to `blockstoredlength`.
function blockstoredlength(a::AbstractArray)
  return storedlength(blocks(a))
end

# BlockArrays

using SparseArraysBase: SparseArraysBase, AbstractSparseArray, AbstractSparseMatrix

_perm(::PermutedDimsArray{<:Any,<:Any,perm}) where {perm} = perm
_invperm(::PermutedDimsArray{<:Any,<:Any,<:Any,invperm}) where {invperm} = invperm
_getindices(t::Tuple, indices) = map(i -> t[i], indices)
_getindices(i::CartesianIndex, indices) = CartesianIndex(_getindices(Tuple(i), indices))

# Represents the array of arrays of a `PermutedDimsArray`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `PermutedDimsArray`.
# TODO: Delete this in favor of `NestedPermutedDimsArrays.NestedPermutedDimsArray`.
struct SparsePermutedDimsArrayBlocks{
  T,N,BlockType<:AbstractArray{T,N},Array<:PermutedDimsArray{T,N}
} <: AbstractSparseArray{BlockType,N}
  array::Array
end
@interface ::AbstractBlockSparseArrayInterface function BlockArrays.blocks(
  a::PermutedDimsArray
)
  return SparsePermutedDimsArrayBlocks{eltype(a),ndims(a),blocktype(parent(a)),typeof(a)}(a)
end
function Base.size(a::SparsePermutedDimsArrayBlocks)
  return _getindices(size(blocks(parent(a.array))), _perm(a.array))
end
function SparseArraysBase.isstored(
  a::SparsePermutedDimsArrayBlocks{<:Any,N}, index::Vararg{Int,N}
) where {N}
  return isstored(blocks(parent(a.array)), _getindices(index, _invperm(a.array))...)
end
function SparseArraysBase.getstoredindex(
  a::SparsePermutedDimsArrayBlocks{<:Any,N}, index::Vararg{Int,N}
) where {N}
  return PermutedDimsArray(
    getstoredindex(blocks(parent(a.array)), _getindices(index, _invperm(a.array))...),
    _perm(a.array),
  )
end
function SparseArraysBase.getunstoredindex(
  a::SparsePermutedDimsArrayBlocks{<:Any,N}, index::Vararg{Int,N}
) where {N}
  return PermutedDimsArray(
    getunstoredindex(blocks(parent(a.array)), _getindices(index, _invperm(a.array))...),
    _perm(a.array),
  )
end
function SparseArraysBase.eachstoredindex(a::SparsePermutedDimsArrayBlocks)
  return map(I -> _getindices(I, _perm(a.array)), eachstoredindex(blocks(parent(a.array))))
end
## TODO: Define `storedvalues` instead.
## function SparseArraysBase.sparse_storage(a::SparsePermutedDimsArrayBlocks)
##   return error("Not implemented")
## end

reverse_index(index) = reverse(index)
reverse_index(index::CartesianIndex) = CartesianIndex(reverse(Tuple(index)))

@interface ::AbstractBlockSparseArrayInterface BlockArrays.blocks(a::Transpose) =
  transpose(blocks(parent(a)))
@interface ::AbstractBlockSparseArrayInterface BlockArrays.blocks(a::Adjoint) =
  adjoint(blocks(parent(a)))

# Represents the array of arrays of a `SubArray`
# wrapping a block spare array, i.e. `blocks(array)` where `a` is a `SubArray`.
struct SparseSubArrayBlocks{T,N,BlockType<:AbstractArray{T,N},Array<:SubArray{T,N}} <:
       AbstractSparseArray{BlockType,N}
  array::Array
end
@interface ::AbstractBlockSparseArrayInterface function BlockArrays.blocks(a::SubArray)
  return SparseSubArrayBlocks{eltype(a),ndims(a),blocktype(parent(a)),typeof(a)}(a)
end
# TODO: Define this as `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
function blockrange(a::SparseSubArrayBlocks)
  blockranges = blockrange.(axes(parent(a.array)), a.array.indices)
  return map(blockrange -> Int.(blockrange), blockranges)
end
function Base.axes(a::SparseSubArrayBlocks)
  return Base.OneTo.(length.(blockrange(a)))
end
function Base.size(a::SparseSubArrayBlocks)
  return length.(axes(a))
end
# TODO: Define `isstored`.
# TODO: Define `getstoredindex`, `getunstoredindex` instead.
function Base.getindex(a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}) where {N}
  # TODO: Should this be defined as `@view a.array[Block(I)]` instead?
  return @view a.array[Block(I)]

  ## parent_blocks = @view blocks(parent(a.array))[blockrange(a)...]
  ## parent_block = parent_blocks[I...]
  ## # TODO: Define this using `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
  ## block = Block(ntuple(i -> blockrange(a)[i][I[i]], ndims(a)))
  ## return @view parent_block[blockindices(parent(a.array), block, a.array.indices)...]
end
# TODO: This should be handled by generic `AbstractSparseArray` code.
# TODO: Define `getstoredindex`, `getunstoredindex` instead.
function Base.getindex(a::SparseSubArrayBlocks{<:Any,N}, I::CartesianIndex{N}) where {N}
  return a[Tuple(I)...]
end
# TODO: Define `setstoredindex!`, `setunstoredindex!` instead.
function Base.setindex!(a::SparseSubArrayBlocks{<:Any,N}, value, I::Vararg{Int,N}) where {N}
  parent_blocks = @view blocks(parent(a.array))[blockrange(a)...]
  # TODO: The following line is required to instantiate
  # uninstantiated blocks, maybe use `@view!` instead,
  # or some other code pattern.
  parent_blocks[I...] = parent_blocks[I...]
  # TODO: Define this using `blockrange(a::AbstractArray, indices::Tuple{Vararg{AbstractUnitRange}})`.
  block = Block(ntuple(i -> blockrange(a)[i][I[i]], ndims(a)))
  return parent_blocks[I...][blockindices(parent(a.array), block, a.array.indices)...] =
    value
end
function Base.isassigned(a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}) where {N}
  if CartesianIndex(I) ∉ CartesianIndices(a)
    return false
  end
  # TODO: Implement this properly.
  return true
end
function SparseArraysBase.eachstoredindex(a::SparseSubArrayBlocks)
  return eachstoredindex(view(blocks(parent(a.array)), blockrange(a)...))
end
# TODO: Either make this the generic interface or define
# `SparseArraysBase.sparse_storage`, which is used
# to defined this.
SparseArraysBase.storedlength(a::SparseSubArrayBlocks) = length(eachstoredindex(a))

## struct SparseSubArrayBlocksStorage{Array<:SparseSubArrayBlocks}
##   array::Array
## end

## TODO: Define `storedvalues` instead.
## function SparseArraysBase.sparse_storage(a::SparseSubArrayBlocks)
##   return map(I -> a[I], eachstoredindex(a))
## end

function SparseArraysBase.getunstoredindex(
  a::SparseSubArrayBlocks{<:Any,N}, I::Vararg{Int,N}
) where {N}
  return error("Not implemented.")
end

to_blocks_indices(I::BlockSlice{<:BlockRange{1}}) = Int.(I.block)
to_blocks_indices(I::BlockIndices{<:Vector{<:Block{1}}}) = Int.(I.blocks)
to_blocks_indices(I::Base.Slice{<:BlockedOneTo}) = Base.OneTo(blocklength(I.indices))

@interface ::AbstractBlockSparseArrayInterface function BlockArrays.blocks(
  a::SubArray{<:Any,<:Any,<:Any,<:Tuple{Vararg{BlockSliceCollection}}}
)
  return @view blocks(parent(a))[map(to_blocks_indices, parentindices(a))...]
end

using BlockArrays: BlocksView
SparseArraysBase.storedlength(a::BlocksView) = length(a)
