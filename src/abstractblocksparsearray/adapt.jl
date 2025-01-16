using Adapt: Adapt, adapt
Adapt.adapt_structure(to, x::AbstractBlockSparseArray) = map_stored_blocks(adapt(to), x)
