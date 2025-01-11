@interface ::AbstractBlockSparseArrayInterface function Base.view(a, I...)
  return Base.invoke(view, Tuple{AbstractArray,Vararg{Any}}, a, I...)
end
