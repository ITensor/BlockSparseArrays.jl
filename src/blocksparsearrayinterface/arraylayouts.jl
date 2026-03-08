using ArrayLayouts: ArrayLayouts, DenseColumnMajor, Dot, MatMulMatAdd, MatMulVecAdd, MulAdd
using BlockArrays: BlockArrays, BlockLayout, muladd!
using LinearAlgebra: LinearAlgebra, dot, mul!
using SparseArraysBase: SparseLayout

const muladd!_blocksparse = blocksparse_style(muladd!)
# Matrix-matrix case
function muladd!_blocksparse(
        α::Number, a1::AbstractMatrix, a2::AbstractMatrix, β::Number, a_dest::AbstractMatrix
    )
    mul!(blocks(a_dest), blocks(a1), blocks(a2), α, β)
    return a_dest
end

# Matrix-vector case: BlockSparseMatrix * dense Vector
function muladd!_blocksparse(
        α::Number, a1::AbstractMatrix, a2::AbstractVector, β::Number, a_dest::AbstractVector
    )
    mul!_blocksparse(a_dest, a1, a2, α, β)
    return a_dest
end

function ArrayLayouts.materialize!(
        m::MatMulMatAdd{
            <:BlockLayout{<:SparseLayout},
            <:BlockLayout{<:SparseLayout},
            <:BlockLayout{<:SparseLayout},
        }
    )
    muladd!_blocksparse(m.α, m.A, m.B, m.β, m.C)
    return m.C
end

# BlockSparseMatrix * dense Vector → dense Vector
function ArrayLayouts.materialize!(
        m::MatMulVecAdd{
            <:BlockLayout{<:SparseLayout},
            <:DenseColumnMajor,
            <:DenseColumnMajor,
        },
    )
    muladd!_blocksparse(m.α, m.A, m.B, m.β, m.C)
    return m.C
end

# BlockSparseMatrix * BlockSparseVector (not yet implemented)
function ArrayLayouts.materialize!(
        m::MatMulVecAdd{
            <:BlockLayout{<:SparseLayout},
            <:BlockLayout{<:SparseLayout},
            <:BlockLayout{<:SparseLayout},
        }
    )
    error("BlockSparseMatrix * BlockSparseVector not implemented.")
    return m.C
end

const dot_blocksparse = blocksparse_style(dot)
function dot_blocksparse(
        a1::AbstractArray, a2::AbstractArray
    )
    # TODO: Add a check that the blocking of `a1` and `a2` are
    # the same, or the same up to a reshape.
    return dot(blocks(a1), blocks(a2))
end

function Base.copy(d::Dot{<:BlockLayout{<:SparseLayout}, <:BlockLayout{<:SparseLayout}})
    return dot_blocksparse(d.A, d.B)
end
