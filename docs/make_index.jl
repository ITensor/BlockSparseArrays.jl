using Literate: Literate
using BlockSparseArrays: BlockSparseArrays

Literate.markdown(
  joinpath(pkgdir(BlockSparseArrays), "examples", "README.jl"),
  joinpath(pkgdir(BlockSparseArrays), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)
