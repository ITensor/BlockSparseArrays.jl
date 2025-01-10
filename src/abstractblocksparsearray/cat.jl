using DerivableInterfaces: @interface, interface

# TODO: Define with `@derive`.
function Base.cat(as::AnyAbstractBlockSparseArray...; dims)
  return @interface interface(as...) cat(as...; dims)
end
