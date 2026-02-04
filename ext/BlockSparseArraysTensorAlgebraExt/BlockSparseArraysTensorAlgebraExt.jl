module BlockSparseArraysTensorAlgebraExt

using BlockArrays: AbstractBlockArray, Block, blocklength, blocks, eachblockaxes1
using BlockSparseArrays: AbstractBlockSparseArray, AbstractBlockSparseMatrix,
    BlockUnitRange, blockrange, blocksparse
using SparseArraysBase: eachstoredindex
using TensorAlgebra: TensorAlgebra, BlockedTuple, matricize, matricize_axes,
    tensor_product_axis, unmatricize

# TODO: Ideally we would use:
# ```julia
# const BlockReshapeFusion = typeof(TensorAlgebra.FusionStyle(AbstractBlockArray))
# ```
# but that doesn't seem to work, i.e. it sometimes returns `ReshapeFusion`. Maybe it is
# a world age issue, though note that `@invokelatest` doesn't seem to fix it.
# For now we just use `Base.get_extension`.
const BlockReshapeFusion =
    Base.get_extension(TensorAlgebra, :TensorAlgebraBlockArraysExt).BlockReshapeFusion

function TensorAlgebra.tensor_product_axis(
        style::BlockReshapeFusion, side::Val{:codomain}, r1::BlockUnitRange, r2::BlockUnitRange,
    )
    return tensor_product_blockrange(style, side, r1, r2)
end
function TensorAlgebra.tensor_product_axis(
        style::BlockReshapeFusion, side::Val{:domain}, r1::BlockUnitRange, r2::BlockUnitRange
    )
    return tensor_product_blockrange(style, side, r1, r2)
end
function tensor_product_blockrange(
        ::BlockReshapeFusion, side::Val, r1::BlockUnitRange, r2::BlockUnitRange
    )
    (isone(first(r1)) && isone(first(r2))) ||
        throw(ArgumentError("Only one-based axes are supported"))
    blockaxpairs = Iterators.product(eachblockaxes1(r1), eachblockaxes1(r2))
    blockaxs = map(blockaxpairs) do (b1, b2)
        # TODO: Store a FusionStyle for the blocks in `BlockReshapeFusion`
        # and use that here.
        return tensor_product_axis(side, b1, b2)
    end
    return blockrange(vec(blockaxs))
end

function TensorAlgebra.matricize(
        style::BlockReshapeFusion, a::AbstractBlockSparseArray, length_codomain::Val
    )
    ax = matricize_axes(style, a, length_codomain)
    reshaped_blocks_a = reshape(blocks(a), blocklength.(ax))
    key(I) = Block(Tuple(I))
    value(I) = matricize(reshaped_blocks_a[I], length_codomain)
    Is = eachstoredindex(reshaped_blocks_a)
    bs = if isempty(Is)
        # Catch empty case and make sure the type is constrained properly.
        # This seems to only be necessary in Julia versions below v1.11,
        # try removing it when we drop support for those versions.
        keytype = Base.promote_op(key, eltype(Is))
        valtype = Base.promote_op(value, eltype(Is))
        valtype′ = !isconcretetype(valtype) ? AbstractMatrix{eltype(a)} : valtype
        Dict{keytype, valtype′}()
    else
        Dict(key(I) => value(I) for I in Is)
    end
    return blocksparse(bs, ax)
end

function TensorAlgebra.unmatricize(
        ::BlockReshapeFusion,
        m::AbstractBlockSparseMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    ax = (codomain_axes..., domain_axes...)
    reshaped_blocks_m = reshape(blocks(m), blocklength.(ax))
    key(I) = Block(Tuple(I))
    function value(I)
        block_axes_I = BlockedTuple(
            map(ntuple(identity, length(ax))) do i
                return Base.axes1(ax[i][Block(I[i])])
            end,
            (length(codomain_axes), length(domain_axes)),
        )
        return unmatricize(reshaped_blocks_m[I], block_axes_I)
    end
    bs = Dict(key(I) => value(I) for I in eachstoredindex(reshaped_blocks_m))
    return blocksparse(bs, ax)
end

end
