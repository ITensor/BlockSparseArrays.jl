var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [BlockSparseArrays]","category":"page"},{"location":"reference/#BlockSparseArrays.BlockSparseArray","page":"Reference","title":"BlockSparseArrays.BlockSparseArray","text":"BlockSparseArray{T}(undef, dims)\nBlockSparseArray{T,N}(undef, dims)\nBlockSparseArray{T,N,A}(undef, dims)\n\nConstruct an uninitialized N-dimensional BlockSparseArray containing elements of type T. dims should be a list of block lengths in each dimension or a list of blocked ranges representing the axes.\n\n\n\n\n\n","category":"type"},{"location":"reference/#BlockSparseArrays.SVD","page":"Reference","title":"BlockSparseArrays.SVD","text":"SVD <: Factorization\n\nMatrix factorization type of the singular value decomposition (SVD) of a matrix A. This is the return type of svd(_), the corresponding matrix factorization function.\n\nIf F::SVD is the factorization object, U, S, V and Vt can be obtained via F.U, F.S, F.V and F.Vt, such that A = U * Diagonal(S) * Vt. The singular values in S are sorted in descending order.\n\nIterating the decomposition produces the components U, S, and V.\n\nExamples\n\njulia> A = [1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]\n4×5 Matrix{Float64}:\n 1.0  0.0  0.0  0.0  2.0\n 0.0  0.0  3.0  0.0  0.0\n 0.0  0.0  0.0  0.0  0.0\n 0.0  2.0  0.0  0.0  0.0\n\njulia> F = BlockSparseArrays.svd(A)\nBlockSparseArrays.SVD{Float64, Float64, Matrix{Float64}, Vector{Float64}, Matrix{Float64}}\nU factor:\n4×4 Matrix{Float64}:\n 0.0  1.0   0.0  0.0\n 1.0  0.0   0.0  0.0\n 0.0  0.0   0.0  1.0\n 0.0  0.0  -1.0  0.0\nsingular values:\n4-element Vector{Float64}:\n 3.0\n 2.23606797749979\n 2.0\n 0.0\nVt factor:\n4×5 Matrix{Float64}:\n -0.0        0.0  1.0  -0.0  0.0\n  0.447214   0.0  0.0   0.0  0.894427\n  0.0       -1.0  0.0   0.0  0.0\n  0.0        0.0  0.0   1.0  0.0\n\njulia> F.U * Diagonal(F.S) * F.Vt\n4×5 Matrix{Float64}:\n 1.0  0.0  0.0  0.0  2.0\n 0.0  0.0  3.0  0.0  0.0\n 0.0  0.0  0.0  0.0  0.0\n 0.0  2.0  0.0  0.0  0.0\n\njulia> u, s, v = F; # destructuring via iteration\n\njulia> u == F.U && s == F.S && v == F.V\ntrue\n\n\n\n\n\n","category":"type"},{"location":"reference/#SparseArraysBase.SparseArrayDOK-Union{Tuple{N}, Tuple{T}, Tuple{BlockArrays.UndefBlocksInitializer, NTuple{N, AbstractUnitRange{<:Integer}}}} where {T, N}","page":"Reference","title":"SparseArraysBase.SparseArrayDOK","text":"SparseArrayDOK{T}(undef_blocks, axes)\nSparseArrayDOK{T,N}(undef_blocks, axes)\n\nConstruct the block structure of an undefined BlockSparseArray that will have blocked axes axes.\n\nNote that undef_blocks is defined in BlockArrays.jl and should be imported from that package to use it as an input to this constructor.\n\n\n\n\n\n","category":"method"},{"location":"reference/#BlockSparseArrays.sparsemortar-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{<:AbstractArray{T, N}, N}, NTuple{N, AbstractUnitRange{<:Integer}}}} where {T, N}","page":"Reference","title":"BlockSparseArrays.sparsemortar","text":"sparsemortar(blocks::AbstractArray{<:AbstractArray{T,N},N}, axes) -> ::BlockSparseArray{T,N}\n\nConstruct a block sparse array from a sparse array of arrays and specified blocked axes. The block sizes must be commensurate with the blocks of the axes.\n\n\n\n\n\n","category":"method"},{"location":"reference/#BlockSparseArrays.svd!-Tuple{Any}","page":"Reference","title":"BlockSparseArrays.svd!","text":"svd!(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD\n\nsvd! is the same as svd, but saves space by overwriting the input A, instead of creating a copy. See documentation of svd for details.\n\n\n\n\n\n","category":"method"},{"location":"reference/#BlockSparseArrays.svd-Tuple{Any}","page":"Reference","title":"BlockSparseArrays.svd","text":"svd(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD\n\nCompute the singular value decomposition (SVD) of A and return an SVD object.\n\nU, S, V and Vt can be obtained from the factorization F with F.U, F.S, F.V and F.Vt, such that A = U * Diagonal(S) * Vt. The algorithm produces Vt and hence Vt is more efficient to extract than V. The singular values in S are sorted in descending order.\n\nIterating the decomposition produces the components U, S, and V.\n\nIf full = false (default), a \"thin\" SVD is returned. For an M times N matrix A, in the full factorization U is M times M and V is N times N, while in the thin factorization U is M times K and V is N times K, where K = min(MN) is the number of singular values.\n\nalg specifies which algorithm and LAPACK method to use for SVD:\n\nalg = DivideAndConquer() (default): Calls LAPACK.gesdd!.\nalg = QRIteration(): Calls LAPACK.gesvd! (typically slower but more accurate) .\n\ncompat: Julia 1.3\nThe alg keyword argument requires Julia 1.3 or later.\n\nExamples\n\njulia> A = rand(4,3);\n\njulia> F = BlockSparseArrays.svd(A); # Store the Factorization Object\n\njulia> A ≈ F.U * Diagonal(F.S) * F.Vt\ntrue\n\njulia> U, S, V = F; # destructuring via iteration\n\njulia> A ≈ U * Diagonal(S) * V'\ntrue\n\njulia> Uonly, = BlockSparseArrays.svd(A); # Store U only\n\njulia> Uonly == U\ntrue\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"../../examples/README.jl\"","category":"page"},{"location":"#BlockSparseArrays.jl","page":"Home","title":"BlockSparseArrays.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Code Style: Blue) (Image: Aqua)","category":"page"},{"location":"","page":"Home","title":"Home","text":"A block sparse array type in Julia based on the BlockArrays.jl interface.","category":"page"},{"location":"#Installation-instructions","page":"Home","title":"Installation instructions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package resides in the ITensor/ITensorRegistry local registry. In order to install, simply add that registry through your package manager. This step is only required once.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg: Pkg\n\njulia> Pkg.Registry.add(url=\"https://github.com/ITensor/ITensorRegistry\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"or:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> Pkg.Registry.add(url=\"git@github.com:ITensor/ITensorRegistry.git\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Then, the package can be added as usual through the package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> Pkg.add(\"BlockSparseArrays\")","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using BlockArrays: Block\nusing BlockSparseArrays: BlockSparseArray, blockstoredlength\nusing Test: @test\n\na = BlockSparseArray{Float64}(undef, [2, 3], [2, 3])\na[Block(1, 2)] = randn(2, 3)\na[Block(2, 1)] = randn(3, 2)\n@test blockstoredlength(a) == 2\nb = a .+ 2 .* a'\n@test Array(b) ≈ Array(a) + 2 * Array(a')\n@test blockstoredlength(b) == 2","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"This page was generated using Literate.jl.","category":"page"}]
}
