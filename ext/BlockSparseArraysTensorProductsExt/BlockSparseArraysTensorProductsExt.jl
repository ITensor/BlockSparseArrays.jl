module BlockSparseArraysTensorProductsExt

using BlockSparseArrays: BlockUnitRange, blockrange, eachblockaxis
using TensorProducts: TensorProducts, tensor_product
# TODO: Dispatch on `FusionStyle` to allow different kinds of products,
# for example to allow merging common symmetry sectors.
function TensorProducts.tensor_product(a1::BlockUnitRange, a2::BlockUnitRange)
  new_blockaxes = vec(
    map(splat(tensor_product), Iterators.product(eachblockaxis(a1), eachblockaxis(a2)))
  )
  return blockrange(new_blockaxes)
end

end
