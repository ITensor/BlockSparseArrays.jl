using BlockArrays:
  AbstractBlockedUnitRange,
  BlockArrays,
  Block,
  BlockIndexRange,
  BlockedVector,
  blocklength,
  blocksize,
  viewblock

# This splits `BlockIndexRange{N}` into
# `NTuple{N,BlockIndexRange{1}}`.
# TODO: Move to `BlockArraysExtensions`.
to_tuple(x) = Tuple(x)
function to_tuple(x::BlockIndexRange{N}) where {N}
  blocks = Tuple(Block(x))
  n = length(blocks)
  return ntuple(dim -> blocks[dim][x.indices[dim]], n)
end

# Override the default definition of `BlockArrays.blocksize`,
# which is incorrect for certain slices.
function BlockArrays.blocksize(a::SubArray{<:Any,<:Any,<:AnyAbstractBlockSparseArray})
  return blocklength.(axes(a))
end
function BlockArrays.blocksize(
  a::SubArray{<:Any,<:Any,<:AnyAbstractBlockSparseArray}, i::Int
)
  # TODO: Maybe use `blocklength(axes(a, i))` which would be a bit faster.
  return blocksize(a)[i]
end

# These definitions circumvent some generic definitions in BlockArrays.jl:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/master/src/views.jl
# which don't handle subslices of blocks properly.
function Base.view(
  a::SubArray{
    <:Any,N,<:AnyAbstractBlockSparseArray,<:Tuple{Vararg{BlockSlice{<:BlockRange{1}},N}}
  },
  I::Block{N},
) where {N}
  return @interface interface(a) view(a, I)
end
function Base.view(
  a::SubArray{
    <:Any,N,<:AnyAbstractBlockSparseArray,<:Tuple{Vararg{BlockSlice{<:BlockRange{1}},N}}
  },
  I::Vararg{Block{1},N},
) where {N}
  return @interface interface(a) view(a, I...)
end
function Base.view(
  V::SubArray{<:Any,1,<:AnyAbstractBlockSparseArray,<:Tuple{BlockSlice{<:BlockRange{1}}}},
  I::Block{1},
)
  return @interface interface(a) view(a, I)
end

# Specialized code for getting the view of a block.
function BlockArrays.viewblock(
  a::AbstractBlockSparseArray{<:Any,N}, block::Block{N}
) where {N}
  return viewblock(a, Tuple(block)...)
end

# TODO: Define `@interface interface(a) viewblock`.
function BlockArrays.viewblock(
  a::AbstractBlockSparseArray{<:Any,N}, block::Vararg{Block{1},N}
) where {N}
  I = CartesianIndex(Int.(block))
  # TODO: isblockstored
  return isstored(blocks(a), I) ? blocks(a)[I] : BlockView(a, block)
end

# Specialized code for getting the view of a subblock.
function Base.view(
  a::AbstractBlockSparseArray{<:Any,N}, block::BlockIndexRange{N}
) where {N}
  return view(a, to_tuple(block)...)
end

# Specialized code for getting the view of a subblock.
function Base.view(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N}}, I::BlockIndexRange{N}
) where {T,N}
  return view(a, to_tuple(I)...)
end
function Base.view(a::AbstractBlockSparseArray{<:Any,N}, I::Vararg{Block{1},N}) where {N}
  return viewblock(a, I...)
end

# TODO: Move to `GradedUnitRanges` or `BlockArraysExtensions`.
to_block(I::Block{1}) = I
to_block(I::BlockIndexRange{1}) = Block(I)
to_block(I::BlockIndexVector{1}) = Block(I)
to_block_indices(I::Block{1}) = Colon()
to_block_indices(I::BlockIndexRange{1}) = only(I.indices)
to_block_indices(I::BlockIndexVector{1}) = only(I.indices)

function Base.view(
  a::AbstractBlockSparseArray{<:Any,N},
  I::Vararg{Union{Block{1},BlockIndexRange{1},BlockIndexVector},N},
) where {N}
  return @views a[to_block.(I)...][to_block_indices.(I)...]
end

function Base.view(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N}}, I::Vararg{Block{1},N}
) where {T,N}
  return viewblock(a, I...)
end
function Base.view(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N}},
  I::Vararg{Union{Block{1},BlockIndexRange{1},BlockIndexVector},N},
) where {T,N}
  return @views a[to_block.(I)...][to_block_indices.(I)...]
end
# Generic fallback.
function BlockArrays.viewblock(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N}}, I::Vararg{Block{1},N}
) where {T,N}
  return Base.invoke(view, Tuple{AbstractArray,Vararg{Any}}, a, I...)
end

function Base.view(
  a::SubArray{
    T,
    N,
    <:AbstractBlockSparseArray{T,N},
    <:Tuple{Vararg{Union{BlockSliceCollection,SubBlockSliceCollection},N}},
  },
  block::Union{Block{N},BlockIndexRange{N}},
) where {T,N}
  return viewblock(a, block)
end

# Fix ambiguity error with BlockArrays.jl for slices like
# `a = BlockSparseArray{Float64}(undef, [2, 2], [2, 2]); @view a[:, :]`.
function Base.view(
  a::SubArray{
    T,
    N,
    <:AbstractBlockSparseArray{T,N},
    <:Tuple{Vararg{Union{Base.Slice,BlockSlice{<:BlockRange{1}}},N}},
  },
  block::Block{N},
) where {T,N}
  return viewblock(a, block)
end

function Base.view(
  a::SubArray{
    T,
    N,
    <:AbstractBlockSparseArray{T,N},
    <:Tuple{Vararg{Union{BlockSliceCollection,SubBlockSliceCollection},N}},
  },
  block::Vararg{Union{Block{1},BlockIndexRange{1}},N},
) where {T,N}
  return viewblock(a, block...)
end
function BlockArrays.viewblock(
  a::SubArray{
    T,
    N,
    <:AbstractBlockSparseArray{T,N},
    <:Tuple{Vararg{Union{BlockSliceCollection,SubBlockSliceCollection},N}},
  },
  block::Union{Block{N},BlockIndexRange{N},BlockIndexVector{N}},
) where {T,N}
  return viewblock(a, to_tuple(block)...)
end

# Fixes ambiguity error with `AnyAbstractBlockSparseArray` definition.
function Base.view(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSlice{<:BlockRange{1}},N}}
  },
  block::Block{N},
) where {T,N}
  return viewblock(a, block)
end
# Fixes ambiguity error with `AnyAbstractBlockSparseArray` definition.
function Base.view(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSlice{<:BlockRange{1}},N}}
  },
  block::Vararg{Block{1},N},
) where {T,N}
  return viewblock(a, block...)
end

# XXX: TODO: Distinguish if a sub-view of the block needs to be taken!
# Define a new `SubBlockSlice` which is used in:
# `@interface interface(a) to_indices(a, inds, I::Tuple{UnitRange{<:Integer},Vararg{Any}})`
# in `blocksparsearrayinterface/blocksparsearrayinterface.jl`.
# TODO: Define `@interface interface(a) viewblock`.
function BlockArrays.viewblock(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockSliceCollection,N}}},
  block::Vararg{Block{1},N},
) where {T,N}
  I = CartesianIndex(Int.(block))
  # TODO: Use `eachblockstoredindex`.
  if I ∈ eachstoredindex(blocks(a))
    return blocks(a)[I]
  end
  return BlockView(parent(a), Block.(Base.reindex(parentindices(blocks(a)), Tuple(I))))
end

function to_blockindexrange(
  a::BlockSlice{<:BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}}}, I::Block{1}
)
  # TODO: Ideally we would just use `a.blocks[I]` but that doesn't
  # work right now.
  return blocks(a.block)[Int(I)]
end
function to_blockindexrange(
  a::BlockIndices{<:BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange}}}, I::Block{1}
)
  # TODO: Ideally we would just use `a.blocks[I]` but that doesn't
  # work right now.
  return blocks(a.blocks)[Int(I)]
end
function to_blockindexrange(
  a::BlockIndices{<:BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexVector}}}, I::Block{1}
)
  # TODO: Ideally we would just use `a.blocks[I]` but that doesn't
  # work right now.
  return blocks(a.blocks)[Int(I)]
end
function to_blockindexrange(
  a::BlockIndices{<:BlockVector{<:GenericBlockIndex{1},<:Vector{<:BlockIndexVector}}},
  I::Block{1},
)
  # TODO: Ideally we would just use `a.blocks[I]` but that doesn't
  # work right now.
  return blocks(a.blocks)[Int(I)]
end
function to_blockindexrange(
  a::Base.Slice{<:AbstractBlockedUnitRange{<:Integer}}, I::Block{1}
)
  @assert I in only(blockaxes(a.indices))
  return I
end

function BlockArrays.viewblock(
  a::SubArray{
    T,
    N,
    <:AbstractBlockSparseArray{T,N},
    <:Tuple{Vararg{Union{BlockSliceCollection,SubBlockSliceCollection},N}},
  },
  block::Vararg{Block{1},N},
) where {T,N}
  brs = ntuple(dim -> to_blockindexrange(parentindices(a)[dim], block[dim]), ndims(a))
  return @view parent(a)[brs...]
end

# TODO: Define `@interface interface() viewblock`.
function BlockArrays.viewblock(
  a::SubArray{
    T,
    N,
    <:AbstractBlockSparseArray{T,N},
    <:Tuple{Vararg{Union{BlockSliceCollection,SubBlockSliceCollection},N}},
  },
  block::Vararg{BlockIndexRange{1},N},
) where {T,N}
  return view(viewblock(a, Block.(block)...), map(b -> only(b.indices), block)...)
end

function Base.view(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{SubBlockSliceCollection,N}}
  },
  block::Union{Block{N},BlockIndexRange{N}},
) where {T,N}
  return viewblock(a, block)
end
function Base.view(
  a::SubArray{T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{BlockIndexRangeSlice,N}}},
  block::Union{Block{N},BlockIndexRange{N}},
) where {T,N}
  return viewblock(a, block)
end
function Base.view(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{SubBlockSliceCollection,N}}
  },
  block::Vararg{Union{Block{1},BlockIndexRange{1},BlockIndexVector},N},
) where {T,N}
  return viewblock(a, block...)
end
function BlockArrays.viewblock(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{SubBlockSliceCollection,N}}
  },
  block::Union{Block{N},BlockIndexRange{N}},
) where {T,N}
  return viewblock(a, to_tuple(block)...)
end

blockedslice_blocks(x::BlockSlice) = x.block
blockedslice_blocks(x::BlockIndices) = x.blocks
# Reinterpret the slice blockwise.
function blockedslice_blocks(x::Base.Slice)
  return mortar(
    map(BlockRange(x.indices)) do b
      return BlockIndexRange(b, Base.Slice(Base.axes1(x.indices[b])))
    end,
  )
end

# TODO: Define `@interface interface(a) viewblock`.
function BlockArrays.viewblock(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{SubBlockSliceCollection,N}}
  },
  I::Vararg{Block{1},N},
) where {T,N}
  # TODO: Use `reindex`, `to_indices`, etc.
  brs = ntuple(ndims(a)) do dim
    # TODO: Ideally we would use this but it outputs a Vector,
    # not a range:
    # return parentindices(a)[dim].block[I[dim]]
    return blocks(blockedslice_blocks(parentindices(a)[dim]))[Int(I[dim])]
  end
  return @view parent(a)[brs...]
end
# TODO: Define `@interface interface(a) viewblock`.
function BlockArrays.viewblock(
  a::SubArray{
    T,N,<:AbstractBlockSparseArray{T,N},<:Tuple{Vararg{SubBlockSliceCollection,N}}
  },
  block::Vararg{BlockIndexRange{1},N},
) where {T,N}
  return view(viewblock(a, Block.(block)...), map(b -> only(b.indices), block)...)
end

# migrate wrapper layer for viewing `adjoint` and `transpose`.
for (f, F) in ((:adjoint, :Adjoint), (:transpose, :Transpose))
  @eval begin
    function Base.view(A::$F{<:Any,<:AbstractBlockSparseVector}, b::Block{1})
      return $f(view(parent(A), b))
    end

    Base.view(A::$F{<:Any,<:AbstractBlockSparseMatrix}, b::Block{2}) = view(A, Tuple(b)...)
    function Base.view(A::$F{<:Any,<:AbstractBlockSparseMatrix}, b1::Block{1}, b2::Block{1})
      return $f(view(parent(A), b2, b1))
    end
  end
end
