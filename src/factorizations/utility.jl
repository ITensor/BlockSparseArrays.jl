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

function blockdiagonalize(A::AbstractBlockSparseMatrix)
  # sort in order to avoid depending on internal details such as dictionary order
  bIs = sort!(collect(eachblockstoredindex(A)); by=Int ∘ last ∘ Tuple)

  # obtain permutation for the blocks that are present
  rowperm = map(first ∘ Tuple, bIs)
  colperm = map(last ∘ Tuple, bIs)

  # These checks might be expensive but are convenient for now
  allunique(rowperm) || throw(ArgumentError("input contains more than one block per row"))
  allunique(colperm) ||
    throw(ArgumentError("input contains more than one block per column"))

  # post-process empty rows and columns: this pairs them up in order of occurance,
  # putting empty rows and columns due to rectangularity at the end
  emptyrows = setdiff(Block.(1:blocksize(A, 1)), rowperm)
  append!(rowperm, emptyrows)
  emptycols = setdiff(Block.(1:blocksize(A, 2)), colperm)
  append!(colperm, emptycols)

  return A[rowperm, colperm], rowperm, colperm
end

function isblockdiagonal(A::AbstractBlockSparseMatrix)
  for bI in eachblockstoredindex(A)
    row, col = Tuple(bI)
    row == col || return false
  end
  # don't need to check for rows and cols appearing only once
  # this is guaranteed as long as eachblockstoredindex is unique, which we assume
  return true
end
