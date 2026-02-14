using BlockArrays: blocksizes
using DiagonalArrays: diagonal
using LinearAlgebra: LinearAlgebra, Diagonal
using MatrixAlgebraKit: MatrixAlgebraKit, default_eig_algorithm, default_eigh_algorithm,
    diagview, eig_full!, eig_vals!, eigh_full!, eigh_vals!

for f in [:default_eig_algorithm, :default_eigh_algorithm]
    @eval begin
        function MatrixAlgebraKit.$f(::Type{<:AbstractBlockSparseMatrix}; kwargs...)
            return BlockDiagonalAlgorithm() do block
                return $f(block; kwargs...)
            end
        end
    end
end

function output_type(::typeof(eig_full!), A::Type{<:AbstractMatrix{T}}) where {T}
    DV = Base.promote_op(eig_full!, A)
    return if isconcretetype(DV)
        DV
    else
        Tuple{AbstractMatrix{complex(T)}, AbstractMatrix{complex(T)}}
    end
end
function output_type(::typeof(eigh_full!), A::Type{<:AbstractMatrix{T}}) where {T}
    DV = Base.promote_op(eigh_full!, A)
    return isconcretetype(DV) ? DV : Tuple{AbstractMatrix{real(T)}, AbstractMatrix{T}}
end

function MatrixAlgebraKit.check_input(
        ::typeof(eig_full!), A::AbstractBlockSparseMatrix, (D, V), ::BlockDiagonalAlgorithm
    )
    @assert isa(D, AbstractBlockSparseMatrix) && isa(V, AbstractBlockSparseMatrix)
    @assert eltype(V) === eltype(D) === complex(eltype(A))
    @assert axes(A, 1) == axes(A, 2)
    @assert axes(A) == axes(D) == axes(V)
    @assert isblockdiagonal(A)
    return nothing
end
function MatrixAlgebraKit.check_input(
        ::typeof(eigh_full!), A::AbstractBlockSparseMatrix, (D, V), ::BlockDiagonalAlgorithm
    )
    @assert isa(D, AbstractBlockSparseMatrix) && isa(V, AbstractBlockSparseMatrix)
    @assert eltype(V) === eltype(A)
    @assert eltype(D) === real(eltype(A))
    @assert axes(A, 1) == axes(A, 2)
    @assert axes(A) == axes(D) == axes(V)
    @assert isblockdiagonal(A)
    return nothing
end

for f in [:eig_full!, :eigh_full!]
    @eval begin
        function MatrixAlgebraKit.initialize_output(
                ::typeof($f), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
            )
            Td, Tv = fieldtypes(output_type($f, blocktype(A)))
            D = similar(A, BlockType(Td))
            V = similar(A, BlockType(Tv))
            return (D, V)
        end
        function MatrixAlgebraKit.$f(
                A::AbstractBlockSparseMatrix, (D, V), alg::BlockDiagonalAlgorithm
            )
            MatrixAlgebraKit.check_input($f, A, (D, V), alg)

            # do decomposition on each block
            for bI in blockdiagindices(A)
                if isstored(A, bI)
                    block = @view!(A[bI])
                    block_alg = block_algorithm(alg, block)
                    bD, bV = $f(block, block_alg)
                    D[bI] = bD
                    V[bI] = bV
                else
                    # TODO: this should be `V[bI] = LinearAlgebra.I`
                    copyto!(@view!(V[bI]), LinearAlgebra.I)
                end
            end
            return (D, V)
        end
    end
end

function output_type(f::typeof(eig_vals!), A::Type{<:AbstractMatrix{T}}) where {T}
    D = Base.promote_op(f, A)
    !isconcretetype(D) && return AbstractVector{complex(T)}
    return D
end
function output_type(f::typeof(eigh_vals!), A::Type{<:AbstractMatrix{T}}) where {T}
    D = Base.promote_op(f, A)
    !isconcretetype(D) && return AbstractVector{real(T)}
    return D
end

for f in [:eig_vals!, :eigh_vals!]
    @eval begin
        function MatrixAlgebraKit.initialize_output(
                ::typeof($f), A::AbstractBlockSparseMatrix, alg::BlockDiagonalAlgorithm
            )
            T = output_type($f, blocktype(A))
            return similar(A, BlockType(T), axes(A, 1))
        end
        function MatrixAlgebraKit.check_input(
                ::typeof($f), A::AbstractBlockSparseMatrix, D, ::BlockDiagonalAlgorithm
            )
            @assert isa(D, AbstractBlockSparseVector)
            @assert eltype(D) === $(f == :eig_vals! ? complex : real)(eltype(A))
            @assert axes(A, 1) == axes(A, 2)
            @assert (axes(A, 1),) == axes(D)
            @assert isblockdiagonal(A)
            return nothing
        end

        function MatrixAlgebraKit.$f(
                A::AbstractBlockSparseMatrix, D, alg::BlockDiagonalAlgorithm
            )
            MatrixAlgebraKit.check_input($f, A, D, alg)
            for I in eachblockstoredindex(A)
                block = @view!(A[I])
                D[Tuple(I)[1]] = $f(block, block_algorithm(alg, block))
            end
            return D
        end
    end
end
