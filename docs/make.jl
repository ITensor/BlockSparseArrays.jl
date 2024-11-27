using BlockSparseArrays: BlockSparseArrays
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  BlockSparseArrays, :DocTestSetup, :(using BlockSparseArrays); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[BlockSparseArrays],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="BlockSparseArrays.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/BlockSparseArrays.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/ITensor/BlockSparseArrays.jl", devbranch="main", push_preview=true
)
