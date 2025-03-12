using DerivableInterfaces: @interface, interface
using DerivableInterfaces.Concatenate: concatenate

function Base._cat(dims, as::AnyAbstractBlockSparseArray...)
  return concatenate(dims, as...)
end
