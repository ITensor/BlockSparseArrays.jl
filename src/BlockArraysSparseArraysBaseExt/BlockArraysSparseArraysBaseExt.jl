using BlockArrays: AbstractBlockArray, BlocksView
using SparseArraysBase: SparseArraysBase, storedlength

function SparseArraysBase.storedlength(a::AbstractBlockArray)
  return sum(b -> storedlength(b), blocks(a); init=zero(Int))
end

# TODO: Handle `BlocksView` wrapping a sparse array?
function SparseArraysBase.eachstoredindex(a::BlocksView)
  return CartesianIndices(a)
end
