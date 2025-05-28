using BlockArrays: BlockArrays, AbstractBlockedUnitRange, Block, blockedrange, blocklasts

struct BlockUnitRange{T,B,CS,R<:AbstractBlockedUnitRange{T,CS}} <:
       AbstractBlockedUnitRange{T,CS}
  r::R
  eachblockaxis::B
end
function blockrange(eachblockaxis)
  return BlockUnitRange(blockedrange(length.(eachblockaxis)), eachblockaxis)
end
Base.first(r::BlockUnitRange) = first(r.r)
Base.last(r::BlockUnitRange) = last(r.r)
BlockArrays.blocklasts(r::BlockUnitRange) = blocklasts(r.r)
eachblockaxis(r::BlockUnitRange) = r.eachblockaxis
function Base.getindex(r::BlockUnitRange, I::Block{1})
  return eachblockaxis(r)[Int(I)] .+ (first(r.r[I]) - 1)
end

function BlockArrays.combine_blockaxes(r1::BlockUnitRange, r2::BlockUnitRange)
  if eachblockaxis(r1) ≠ eachblockaxis(r2)
    return throw(ArgumentError("BlockUnitRanges must have the same block axes"))
  end
  return r1
end
