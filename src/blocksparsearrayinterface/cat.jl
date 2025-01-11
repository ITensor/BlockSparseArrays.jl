using BlockArrays: AbstractBlockedUnitRange, blockedrange, blocklengths
using DerivableInterfaces: DerivableInterfaces, @interface, cat!
using SparseArraysBase: SparseArraysBase

# TODO: Maybe move to `DerivableInterfacesBlockArraysExt`.
# TODO: Handle dual graded unit ranges, for example in a new `SparseArraysBaseGradedUnitRangesExt`.
function DerivableInterfaces.axis_cat(
  a1::AbstractBlockedUnitRange, a2::AbstractBlockedUnitRange
)
  return blockedrange(vcat(blocklengths(a1), blocklengths(a2)))
end

@interface ::AbstractBlockSparseArrayInterface function DerivableInterfaces.cat!(
  a_dest::AbstractArray, as::AbstractArray...; dims
)
  cat!(blocks(a_dest), blocks.(as)...; dims)
  return a_dest
end
