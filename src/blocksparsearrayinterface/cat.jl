using BlockArrays: blocks
using DerivableInterfaces.Concatenate: Concatenated, cat!

function Base.copyto!(
  dest::AbstractArray, concat::Concatenated{<:BlockSparseArrayInterface}
)
  # TODO: This assumes the destination blocking is commensurate with
  # the blocking of the sources, for example because it was constructed
  # based on the input arguments. Maybe check that explicitly.
  # This should mostly just get called from `cat` anyway and not get
  # called explicitly.
  cat!(blocks(dest), blocks.(concat.args)...; dims=concat.dims)
  return dest
end
