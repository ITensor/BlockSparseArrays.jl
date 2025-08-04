function infimum(r1::AbstractUnitRange, r2::AbstractUnitRange)
  (isone(first(r1)) && isone(first(r2))) ||
    throw(ArgumentError("infimum only defined for ranges starting at 1"))
  if length(r1) ≤ length(r2)
    return r1
  else
    return r1[r2]
  end
end

function supremum(r1::AbstractUnitRange, r2::AbstractUnitRange)
  (isone(first(r1)) && isone(first(r2))) ||
    throw(ArgumentError("supremum only defined for ranges starting at 1"))
  if length(r1) ≥ length(r2)
    return r1
  else
    return r2
  end
end
