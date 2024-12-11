using BlockArrays: Block
using SparseArraysBase: SparseArraysBase, eachstoredindex, storedlength, storedvalues

# Structure storing the block sparse storage
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

## TODO: Delete this, define `getstoredindex`, etc.
## function SparseArraysBase.sparse_storage(a::AbstractBlockSparseArray)
##   return BlockSparseStorage(a)
## end

function SparseArraysBase.storedlength(a::AnyAbstractBlockSparseArray)
  return sum(storedlength, storedvalues(blocks(a)); init=zero(Int))
end
