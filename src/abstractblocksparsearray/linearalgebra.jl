using LinearAlgebra: LinearAlgebra, Adjoint, Transpose, norm, tr

# Like: https://github.com/JuliaLang/julia/blob/v1.11.1/stdlib/LinearAlgebra/src/transpose.jl#L184
# but also takes the dual of the axes.
# Fixes an issue raised in:
# https://github.com/ITensor/ITensors.jl/issues/1336#issuecomment-2353434147
function Base.copy(a::Adjoint{T,<:AbstractBlockSparseMatrix{T}}) where {T}
  a_dest = similar(parent(a), axes(a))
  a_dest .= a
  return a_dest
end

# More efficient than the generic `LinearAlgebra` version.
function Base.copy(a::Transpose{T,<:AbstractBlockSparseMatrix{T}}) where {T}
  a_dest = similar(parent(a), axes(a))
  a_dest .= a
  return a_dest
end

function LinearAlgebra.norm(a::AnyAbstractBlockSparseArray, p::Real=2)
  nrmᵖ = float(norm(zero(eltype(a))))
  for I in eachblockstoredindex(a)
    nrmᵖ += norm(@view(a[I]), p)^p
  end
  return nrmᵖ^(1/p)
end

function LinearAlgebra.tr(a::AnyAbstractBlockSparseMatrix)
  tr_a = zero(eltype(a))
  for I in eachstoredblockdiagindex(a)
    tr_a += tr(@view(a[I]))
  end
  return tr_a
end
