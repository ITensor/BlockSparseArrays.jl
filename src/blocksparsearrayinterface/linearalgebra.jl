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
