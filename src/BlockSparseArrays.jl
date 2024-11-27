module BlockSparseArrays
include("BlockArraysExtensions/BlockArraysExtensions.jl")
include("blocksparsearrayinterface/blocksparsearrayinterface.jl")
include("blocksparsearrayinterface/linearalgebra.jl")
include("blocksparsearrayinterface/blockzero.jl")
include("blocksparsearrayinterface/broadcast.jl")
include("blocksparsearrayinterface/map.jl")
include("blocksparsearrayinterface/arraylayouts.jl")
include("blocksparsearrayinterface/views.jl")
include("blocksparsearrayinterface/cat.jl")
include("abstractblocksparsearray/abstractblocksparsearray.jl")
include("abstractblocksparsearray/wrappedabstractblocksparsearray.jl")
include("abstractblocksparsearray/abstractblocksparsematrix.jl")
include("abstractblocksparsearray/abstractblocksparsevector.jl")
include("abstractblocksparsearray/views.jl")
include("abstractblocksparsearray/arraylayouts.jl")
include("abstractblocksparsearray/sparsearrayinterface.jl")
include("abstractblocksparsearray/broadcast.jl")
include("abstractblocksparsearray/map.jl")
include("abstractblocksparsearray/linearalgebra.jl")
include("abstractblocksparsearray/cat.jl")
include("blocksparsearray/defaults.jl")
include("blocksparsearray/blocksparsearray.jl")
include("BlockArraysSparseArraysBaseExt/BlockArraysSparseArraysBaseExt.jl")
include("../ext/BlockSparseArraysTensorAlgebraExt/src/BlockSparseArraysTensorAlgebraExt.jl")
include(
  "../ext/BlockSparseArraysGradedUnitRangesExt/src/BlockSparseArraysGradedUnitRangesExt.jl"
)
include("../ext/BlockSparseArraysAdaptExt/src/BlockSparseArraysAdaptExt.jl")
end
