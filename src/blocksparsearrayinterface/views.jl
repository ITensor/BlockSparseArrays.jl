const view_blocksparse = blocksparse_style(view)
function view_blocksparse(a, I...)
    return Base.invoke(view, Tuple{AbstractArray, Vararg{Any}}, a, I...)
end
