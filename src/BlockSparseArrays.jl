module BlockSparseArrays

export BlockSparseArray,
  BlockSparseMatrix,
  BlockSparseVector,
  blockstoredlength,
  eachblockstoredindex,
  eachstoredblock,
  sparsemortar

# factorizations
include("factorizations/svd.jl")

# possible upstream contributions
include("BlockArraysExtensions/BlockArraysExtensions.jl")

# interface functions that don't have to specialize
include("blocksparsearrayinterface/blocksparsearrayinterface.jl")
include("blocksparsearrayinterface/linearalgebra.jl")
include("blocksparsearrayinterface/getunstoredblock.jl")
include("blocksparsearrayinterface/broadcast.jl")
include("blocksparsearrayinterface/map.jl")
include("blocksparsearrayinterface/arraylayouts.jl")
include("blocksparsearrayinterface/views.jl")
include("blocksparsearrayinterface/cat.jl")

# functions defined for any abstractblocksparsearray
include("abstractblocksparsearray/abstractblocksparsearray.jl")
include("abstractblocksparsearray/abstractblocksparsematrix.jl")
include("abstractblocksparsearray/abstractblocksparsevector.jl")
include("abstractblocksparsearray/wrappedabstractblocksparsearray.jl")
include("abstractblocksparsearray/views.jl")
include("abstractblocksparsearray/arraylayouts.jl")
include("abstractblocksparsearray/sparsearrayinterface.jl")
include("abstractblocksparsearray/broadcast.jl")
include("abstractblocksparsearray/map.jl")
include("abstractblocksparsearray/linearalgebra.jl")
include("abstractblocksparsearray/cat.jl")
include("abstractblocksparsearray/adapt.jl")

# functions specifically for BlockSparseArray
include("blocksparsearray/blocksparsearray.jl")
include("blocksparsearray/blockdiagonalarray.jl")

include("BlockArraysSparseArraysBaseExt/BlockArraysSparseArraysBaseExt.jl")

end
