using DerivableInterfaces: @interface, interface

function Base._cat(dims, as::AnyAbstractBlockSparseArray...)
  # TODO: Call `DerivableInterfaces.cat_along(dims, as...)` instead,
  # for better inferability. See:
  # https://github.com/ITensor/DerivableInterfaces.jl/pull/13
  # https://github.com/ITensor/DerivableInterfaces.jl/pull/17
  return @interface interface(as...) cat(as...; dims)
end
