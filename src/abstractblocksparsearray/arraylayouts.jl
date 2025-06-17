using ArrayLayouts: ArrayLayouts, DualLayout, MemoryLayout, MulAdd
using BlockArrays: BlockLayout
using SparseArraysBase: SparseLayout
using TypeParameterAccessors: parenttype, similartype

function ArrayLayouts.MemoryLayout(arraytype::Type{<:AnyAbstractBlockSparseArray})
  outer_layout = typeof(MemoryLayout(blockstype(arraytype)))
  inner_layout = typeof(MemoryLayout(blocktype(arraytype)))
  return BlockLayout{outer_layout,inner_layout}()
end

# TODO: Generalize to `BlockSparseVectorLike`/`AnyBlockSparseVector`.
function ArrayLayouts.MemoryLayout(
  arraytype::Type{<:Adjoint{<:Any,<:AbstractBlockSparseVector}}
)
  return DualLayout{typeof(MemoryLayout(parenttype(arraytype)))}()
end
# TODO: Generalize to `BlockSparseVectorLike`/`AnyBlockSparseVector`.
function ArrayLayouts.MemoryLayout(
  arraytype::Type{<:Transpose{<:Any,<:AbstractBlockSparseVector}}
)
  return DualLayout{typeof(MemoryLayout(parenttype(arraytype)))}()
end

function Base.similar(
  mul::MulAdd{
    <:BlockLayout{<:SparseLayout,BlockLayoutA},
    <:BlockLayout{<:SparseLayout,BlockLayoutB},
    LayoutC,
    T,
    A,
    B,
    C,
  },
  elt::Type,
  axes,
) where {BlockLayoutA,BlockLayoutB,LayoutC,T,A,B,C}

  # TODO: Consider using this instead:
  # ```julia
  # blockmultype = MulAdd{BlockLayoutA,BlockLayoutB,LayoutC,T,blocktype(A),blocktype(B),C}
  # output_blocktype = Base.promote_op(
  #   similar, blockmultype, Type{elt}, Tuple{eltype.(eachblockaxis.(axes))...}
  # )
  # ```
  # The issue is that it in some cases it seems to lose some information about the block types.

  # TODO: Maybe this should be:
  # output_blocktype = Base.promote_op(
  #   mul!, blocktype(mul.A), blocktype(mul.B), blocktype(mul.C), typeof(mul.α), typeof(mul.β)
  # )

  output_blocktype = Base.promote_op(*, blocktype(mul.A), blocktype(mul.B))
  output_blocktype′ =
    !isconcretetype(output_blocktype) ? AbstractMatrix{elt} : output_blocktype
  return similar(BlockSparseArray{elt,length(axes),output_blocktype′}, axes)
end

# Materialize a SubArray view.
function ArrayLayouts.sub_materialize(layout::BlockLayout{<:SparseLayout}, a, axes)
  # TODO: Define `blocktype`/`blockstype` for `SubArray` wrapping `BlockSparseArray`.
  # TODO: Use `similar`?
  blocktype_a = blocktype(parent(a))
  a_dest = BlockSparseArray{eltype(a),length(axes),blocktype_a}(undef, axes)
  a_dest .= a
  return a_dest
end

function _similar(arraytype::Type{<:AbstractArray}, size::Tuple)
  return similar(arraytype, size)
end
function _similar(
  ::Type{<:SubArray{<:Any,<:Any,<:ArrayType}}, size::Tuple
) where {ArrayType}
  return similar(ArrayType, size)
end

# Materialize a SubArray view.
function ArrayLayouts.sub_materialize(
  layout::BlockLayout{<:SparseLayout}, a, axes::Tuple{Vararg{Base.OneTo}}
)
  a_dest = _similar(blocktype(a), length.(axes))
  a_dest .= a
  return a_dest
end
