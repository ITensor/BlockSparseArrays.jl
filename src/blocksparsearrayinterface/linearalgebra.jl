using BlockArrays: Block
using LinearAlgebra: LinearAlgebra, mul!

const mul!_blocksparse = blocksparse_style(mul!)
function mul!_blocksparse(
        a_dest::AbstractMatrix,
        a1::AbstractMatrix,
        a2::AbstractMatrix,
        α::Number = true,
        β::Number = false,
    )
    mul!(blocks(a_dest), blocks(a1), blocks(a2), α, β)
    return a_dest
end

# Matrix-vector multiplication: BlockSparseMatrix * dense Vector
function mul!_blocksparse(
        a_dest::AbstractVector,
        a1::AbstractMatrix,
        a2::AbstractVector,
        α::Number = true,
        β::Number = false,
    )
    # Scale destination by β (or zero it if β == 0)
    if iszero(β)
        fill!(a_dest, zero(eltype(a_dest)))
    elseif !isone(β)
        a_dest .*= β
    end

    # Accumulate A[i,j] * v[j_range] into C[i_range]
    for I in eachblockstoredindex(a1)
        i, j = Int.(Tuple(I))
        block_A = a1[I]
        row_range = axes(a1, 1)[Block(i)]
        col_range = axes(a1, 2)[Block(j)]
        v_block = @view a2[col_range]
        c_block = @view a_dest[row_range]
        mul!(c_block, block_A, v_block, α, true)  # β=true to accumulate
    end
    return a_dest
end
