using FunctionImplementations.Concatenate: concatenate

function Base._cat(dims, as::AnyAbstractBlockSparseArray...)
    return concatenate(dims, as...)
end
