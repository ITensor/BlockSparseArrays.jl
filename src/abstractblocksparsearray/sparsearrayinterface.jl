using BlockArrays: Block
using SparseArraysBase: SparseArraysBase, eachstoredindex, storedlength, storedvalues

# Structure storing the block sparse storage
# TODO: Delete this in favor of `storedvalues(blocks(a))`,
# and rename `storedblocks(a)` and/or `eachstoredblock(a)`.
struct BlockSparseStorage{Arr<:AbstractBlockSparseArray}
  array::Arr
end

function blockindex_to_cartesianindex(a::AbstractArray, blockindex)
  return CartesianIndex(getindex.(axes(a), getindex.(Block.(blockindex.I), blockindex.α)))
end

function Base.keys(s::BlockSparseStorage)
  stored_blockindices = Iterators.map(eachstoredindex(blocks(s.array))) do I
    block_axes = axes(blocks(s.array)[I])
    blockindices = Block(Tuple(I))[block_axes...]
    return Iterators.map(
      blockindex -> blockindex_to_cartesianindex(s.array, blockindex), blockindices
    )
  end
  return Iterators.flatten(stored_blockindices)
end

function Base.values(s::BlockSparseStorage)
  return Iterators.map(I -> s.array[I], eachindex(s))
end

function Base.iterate(s::BlockSparseStorage, args...)
  return iterate(values(s), args...)
end

## TODO: Bring back this deifinition but check that it makes sense.
## function SparseArraysBase.storedvaluese(a::AbstractBlockSparseArray)
##   return BlockSparseStorage(a)
## end

# TODO: Turn this into an `@interface ::AbstractBlockSparseArrayInterface` function.
function SparseArraysBase.storedlength(a::AnyAbstractBlockSparseArray)
  return sum(storedlength, storedvalues(blocks(a)); init=zero(Int))
end
