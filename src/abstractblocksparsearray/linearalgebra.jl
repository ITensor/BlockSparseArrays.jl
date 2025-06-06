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

# TODO: Define `SparseArraysBase.isdiag`, define as
# `isdiag(blocks(a))`.
function blockisdiag(a::AbstractArray)
  return all(eachblockstoredindex(a)) do I
    return allequal(Tuple(I))
  end
end

const MATRIX_FUNCTIONS = [
  :exp,
  :cis,
  :log,
  :sqrt,
  :cbrt,
  :cos,
  :sin,
  :tan,
  :csc,
  :sec,
  :cot,
  :cosh,
  :sinh,
  :tanh,
  :csch,
  :sech,
  :coth,
  :acos,
  :asin,
  :atan,
  :acsc,
  :asec,
  :acot,
  :acosh,
  :asinh,
  :atanh,
  :acsch,
  :asech,
  :acoth,
]

# Functions where the dense implementations in `LinearAlgebra` are
# not type stable.
const MATRIX_FUNCTIONS_UNSTABLE = [
  :log,
  :sqrt,
  :acos,
  :asin,
  :atan,
  :acsc,
  :asec,
  :acot,
  :acosh,
  :asinh,
  :atanh,
  :acsch,
  :asech,
  :acoth,
]

function initialize_output_blocksparse(f::F, a::AbstractMatrix) where {F}
  B = Base.promote_op(f, blocktype(a))
  return similar(a, BlockType(B))
end

function matrix_function_blocksparse(f::F, a::AbstractMatrix; kwargs...) where {F}
  blockisdiag(a) || throw(ArgumentError("`$f` only defined for block-diagonal matrices"))
  fa = initialize_output_blocksparse(f, a)
  for I in blockdiagindices(a)
    fa[I] = f(a[I]; kwargs...)
  end
  return fa
end

for f in MATRIX_FUNCTIONS
  @eval begin
    function Base.$f(a::AnyAbstractBlockSparseMatrix)
      return matrix_function_blocksparse($f, a)
    end
  end
end

for f in MATRIX_FUNCTIONS_UNSTABLE
  @eval begin
    function initialize_output_blocksparse(::typeof($f), a::AbstractMatrix)
      B = similartype(blocktype(a), complex(eltype(a)))
      return similar(a, BlockType(B))
    end
  end
end

function LinearAlgebra.inv(a::AnyAbstractBlockSparseMatrix)
  return matrix_function_blocksparse(inv, a)
end

using LinearAlgebra: LinearAlgebra, pinv
function LinearAlgebra.pinv(a::AnyAbstractBlockSparseMatrix; kwargs...)
  return matrix_function_blocksparse(pinv, a; kwargs...)
end
