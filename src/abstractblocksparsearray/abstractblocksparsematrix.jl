const AbstractBlockSparseMatrix{T} = AbstractBlockSparseArray{T,2}

# SVD is implemented by trying to
# 1. Attempt to find a block-diagonal implementation by permuting
# 2. Fallback to AbstractBlockArray implementation via BlockedArray

function eigencopy_oftype(A::AbstractBlockSparseMatrix, T)
  if is_block_permutation_matrix(A)
    Acopy = similar(A, T)
    for bI in eachblockstoredindex(A)
      Acopy[bI] = eigencopy_oftype(A[bI], T)
    end
    return Acopy
  else
    return BlockedMatrix{T}(A)
  end
end

function is_block_permutation_matrix(a::AbstractBlockSparseMatrix)
  return allunique(first ∘ Tuple, eachblockstoredindex(a)) &&
         allunique(last ∘ Tuple, eachblockstoredindex(a))
end

function _allocate_svd_output(A::AbstractBlockSparseMatrix, full::Bool, ::Algorithm)
  @assert !full "TODO"
  bm, bn = blocksize(A)
  bmn = min(bm, bn)

  brows = blocklengths(axes(A, 1))
  bcols = blocklengths(axes(A, 2))
  slengths = Vector{Int}(undef, bmn)

  # fill in values for blocks that are present
  bIs = collect(eachblockstoredindex(A))
  browIs = Int.(first.(Tuple.(bIs)))
  bcolIs = Int.(last.(Tuple.(bIs)))
  for bI in eachblockstoredindex(A)
    row, col = Int.(Tuple(bI))
    nrows = brows[row]
    ncols = bcols[col]
    slengths[col] = min(nrows, ncols)
  end

  # fill in values for blocks that aren't present, pairing them in order of occurence
  # this is a convention, which at least gives the expected results for blockdiagonal
  emptyrows = findall(∉(browIs), 1:bmn)
  emptycols = findall(∉(bcolIs), 1:bmn)
  for (row, col) in zip(emptyrows, emptycols)
    slengths[col] = min(brows[row], bcols[col])
  end

  U = similar(A, axes(A, 1), blockedrange(slengths))
  S = similar(A, real(eltype(A)), blockedrange(slengths))
  Vt = similar(A, blockedrange(slengths), axes(A, 2))

  return U, S, Vt
end

function svd(A::AbstractBlockSparseMatrix; kwargs...)
  return svd!(eigencopy_oftype(A, LinearAlgebra.eigtype(eltype(A))); kwargs...)
end

function svd!(
  A::AbstractBlockSparseMatrix; full::Bool=false, alg::Algorithm=default_svd_alg(A)
)
  @assert is_block_permutation_matrix(A) "Cannot keep sparsity: use `svd` to convert to `BlockedMatrix"
  U, S, Vt = _allocate_svd_output(A, full, alg)
  for bI in eachblockstoredindex(A)
    bUSV = svd!(A[bI]; full, alg)
    brow, bcol = Tuple(bI)
    U[brow, bcol] = bUSV.U
    S[bcol] = bUSV.S
    Vt[bcol, bcol] = bUSV.Vt
  end
  return SVD(U, S, Vt)
end
