using BlockSparseArrays: BlockSparseArrays
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  BlockSparseArrays,
  :DocTestSetup,
  quote
    using BlockSparseArrays
    using LinearAlgebra: Diagonal
  end;
  recursive=true,
)

include("make_index.jl")

makedocs(;
  modules=[BlockSparseArrays],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="BlockSparseArrays.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/BlockSparseArrays.jl",
    edit_link="main",
    assets=["assets/favicon.ico", "assets/extras.css"],
  ),
  pages=["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo="github.com/ITensor/BlockSparseArrays.jl", devbranch="main", push_preview=true
)
