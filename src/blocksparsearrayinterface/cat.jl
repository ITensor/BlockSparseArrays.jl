using BlockArrays: AbstractBlockedUnitRange, blockedrange, blocklengths
using Derive: Derive, @interface, cat!
using SparseArraysBase: SparseArraysBase

# TODO: Maybe move to `DeriveBlockArraysExt`.
# TODO: Handle dual graded unit ranges, for example in a new `SparseArraysBaseGradedUnitRangesExt`.
function Derive.axis_cat(a1::AbstractBlockedUnitRange, a2::AbstractBlockedUnitRange)
  return blockedrange(vcat(blocklengths(a1), blocklengths(a2)))
end

@interface ::AbstractBlockSparseArrayInterface function Derive.cat!(
  a_dest::AbstractArray, as::AbstractArray...; dims
)
  cat!(blocks(a_dest), blocks.(as)...; dims)
  return a_dest
end
