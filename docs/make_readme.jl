using Literate: Literate
using BlockSparseArrays: BlockSparseArrays

Literate.markdown(
  joinpath(pkgdir(BlockSparseArrays), "examples", "README.jl"),
  joinpath(pkgdir(BlockSparseArrays));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
