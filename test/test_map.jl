using Adapt: adapt
using BlockArrays: Block, BlockRange, blockedrange, blocksize, blocksizes, mortar
using BlockSparseArrays:
  BlockSparseArray,
  BlockSparseVector,
  BlockVector,
  BlockView,
  BlockedVector,
  blocklengths,
  blockstoredlength,
  blocktype,
  eachblockstoredindex
using DerivableInterfaces: zero!
using GPUArraysCore: @allowscalar
using JLArrays: JLArray
using SparseArraysBase: storedlength
using Test: @test, @test_broken, @test_throws, @testset

elts = (Float32, Float64, ComplexF32)
arrayts = (Array, JLArray)
@testset "map and slicing (arraytype=$arrayt, eltype=$elt)" for arrayt in arrayts,
  elt in elts

  dev = adapt(arrayt)

  a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = dev(randn(elt, size(a[b])))
  end
  @test eltype(a) == elt
  @test blockstoredlength(a) == 2
  @test storedlength(a) == 2 * 4 + 3 * 3

  # TODO: Broken on GPU.
  if arrayt ≠ Array
    a = dev(BlockSparseArray{elt}(undef, [2, 3], [3, 4]))
    @test_broken a[Block(1, 2)] .= 2
  end

  # TODO: Broken on GPU.
  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  a[Block(1, 2)] .= 2
  @test eltype(a) == elt
  @test all(==(2), a[Block(1, 2)])
  @test iszero(a[Block(1, 1)])
  @test iszero(a[Block(2, 1)])
  @test iszero(a[Block(2, 2)])
  @test blockstoredlength(a) == 1
  @test storedlength(a) == 2 * 4

  # TODO: Broken on GPU.
  if arrayt ≠ Array
    a = dev(BlockSparseArray{elt}(undef, [2, 3], [3, 4]))
    @test_broken a[Block(1, 2)] .= 0
  end

  # TODO: Broken on GPU.
  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  a[Block(1, 2)] .= 0
  @test eltype(a) == elt
  @test iszero(a[Block(1, 1)])
  @test iszero(a[Block(2, 1)])
  @test iszero(a[Block(1, 2)])
  @test iszero(a[Block(2, 2)])
  @test blockstoredlength(a) == 1
  @test storedlength(a) == 2 * 4

  # Test similar on broadcasted expressions.
  a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
  bc = Broadcast.broadcasted(+, a, a)
  a′ = similar(bc, Float32)
  @test a′ isa BlockSparseArray{Float32}
  @test blocktype(a′) <: arrayt{Float32,2}
  @test axes(a) == (blockedrange([2, 3]), blockedrange([3, 4]))

  # Test similar on broadcasted expressions with axes specified.
  a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
  bc = Broadcast.broadcasted(+, a, a)
  a′ = similar(
    bc, Float32, (blockedrange([2, 4]), blockedrange([2, 5]), blockedrange([2, 2]))
  )
  @test a′ isa BlockSparseArray{Float32}
  @test blocktype(a′) <: arrayt{Float32,3}
  @test axes(a′) == (blockedrange([2, 4]), blockedrange([2, 5]), blockedrange([2, 2]))

  a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = dev(randn(elt, size(a[b])))
  end
  b = similar(a, complex(elt))
  @test eltype(b) == complex(eltype(a))
  @test iszero(b)
  @test blockstoredlength(b) == 0
  @test storedlength(b) == 0
  @test size(b) == size(a)
  @test blocksize(b) == blocksize(a)

  # Regression test for https://github.com/ITensor/BlockSparseArrays.jl/issues/98
  a = dev(BlockSparseArray{elt}(undef))
  b = similar(a, Float64, (Base.OneTo(1),))
  @test b isa BlockSparseVector{Float64}
  @test size(b) == (1,)
  @test blocksize(b) == (1,)

  a = dev(BlockSparseArray{elt}(undef, [2, 3], [3, 4]))
  b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
  c = @view b[Block(1, 1)]
  @test iszero(a)
  @test iszero(storedlength(a))
  @test iszero(b)
  @test iszero(storedlength(b))
  # TODO: Broken on GPU.
  @test iszero(c) broken = arrayt ≠ Array
  @test iszero(storedlength(c))
  @allowscalar a[5, 7] = 1
  @test !iszero(a)
  @test storedlength(a) == 3 * 4
  @test !iszero(b)
  @test storedlength(b) == 3 * 4
  # TODO: Broken on GPU.
  @test !iszero(c) broken = arrayt ≠ Array
  @test storedlength(c) == 3 * 4
  d = @view a[1:4, 1:6]
  @test iszero(d)
  @test storedlength(d) == 2 * 3

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = copy(a)
  b[1, 1] = 11
  @test b[1, 1] == 11
  @test a[1, 1] ≠ 11

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = copy(a)
  b .*= 2
  @test b ≈ 2a

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = copy(a)
  b ./= 2
  @test b ≈ a / 2

  a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = dev(randn(elt, size(a[b])))
  end
  b = 2 * a
  @allowscalar @test Array(b) ≈ 2 * Array(a)
  @test eltype(b) == elt
  @test blockstoredlength(b) == 2
  @test storedlength(b) == 2 * 4 + 3 * 3

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = (2 + 3im) * a
  @test Array(b) ≈ (2 + 3im) * Array(a)
  @test eltype(b) == complex(elt)
  @test blockstoredlength(b) == 2
  @test storedlength(b) == 2 * 4 + 3 * 3

  a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = dev(randn(elt, size(a[b])))
  end
  b = a + a
  @allowscalar @test Array(b) ≈ 2 * Array(a)
  @test eltype(b) == elt
  @test blockstoredlength(b) == 2
  @test storedlength(b) == 2 * 4 + 3 * 3

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  x = BlockSparseArray{elt}(undef, ([3, 4], [2, 3]))
  @views for b in [Block(1, 2), Block(2, 1)]
    x[b] = randn(elt, size(x[b]))
  end
  b = a .+ a .+ 3 .* PermutedDimsArray(x, (2, 1))
  @test Array(b) ≈ 2 * Array(a) + 3 * permutedims(Array(x), (2, 1))
  @test eltype(b) == elt
  @test blockstoredlength(b) == 2
  @test storedlength(b) == 2 * 4 + 3 * 3

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = permutedims(a, (2, 1))
  @test Array(b) ≈ permutedims(Array(a), (2, 1))
  @test eltype(b) == elt
  @test blockstoredlength(b) == 2
  @test storedlength(b) == 2 * 4 + 3 * 3

  a = dev(BlockSparseArray{elt}(undef, [1, 1, 1], [1, 2, 3], [2, 2, 1], [1, 2, 1]))
  a[Block(3, 2, 2, 3)] = dev(randn(elt, 1, 2, 2, 1))
  perm = (2, 3, 4, 1)
  for b in (PermutedDimsArray(a, perm), permutedims(a, perm))
    @test @allowscalar(Array(b)) == permutedims(Array(a), perm)
    @test issetequal(eachblockstoredindex(b), [Block(2, 2, 3, 3)])
    @test @allowscalar b[Block(2, 2, 3, 3)] == permutedims(a[Block(3, 2, 2, 3)], perm)
  end

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = map(x -> 2x, a)
  @test Array(b) ≈ 2 * Array(a)
  @test eltype(b) == elt
  @test size(b) == size(a)
  @test blocksize(b) == (2, 2)
  @test blockstoredlength(b) == 2
  @test storedlength(b) == 2 * 4 + 3 * 3

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = a[[Block(2), Block(1)], [Block(2), Block(1)]]
  @test b[Block(1, 1)] == a[Block(2, 2)]
  @test b[Block(1, 2)] == a[Block(2, 1)]
  @test b[Block(2, 1)] == a[Block(1, 2)]
  @test b[Block(2, 2)] == a[Block(1, 1)]
  @test size(b) == size(a)
  @test blocksize(b) == (2, 2)
  @test storedlength(b) == storedlength(a)
  @test blockstoredlength(b) == 2

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = a[Block(1):Block(2), Block(1):Block(2)]
  @test b == a
  @test size(b) == size(a)
  @test blocksize(b) == (2, 2)
  @test storedlength(b) == storedlength(a)
  @test blockstoredlength(b) == 2

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = a[Block(1):Block(1), Block(1):Block(2)]
  @test b == Array(a)[1:2, 1:end]
  @test b[Block(1, 1)] == a[Block(1, 1)]
  @test b[Block(1, 2)] == a[Block(1, 2)]
  @test size(b) == (2, 7)
  @test blocksize(b) == (1, 2)
  @test storedlength(b) == storedlength(a[Block(1, 2)])
  @test blockstoredlength(b) == 1

  a = dev(BlockSparseArray{elt}(undef, ([2, 3], [3, 4])))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = dev(randn(elt, size(a[b])))
  end
  I = 2:4
  I′, J′ = falses.(size(a))
  I′[I] .= true
  J′[I] .= true
  for b in (a[I, I], @view(a[I, I]), a[I′, J′], @view(a[I′, J′]))
    @allowscalar @test b == Array(a)[2:4, 2:4]
    @test size(b) == (3, 3)
    @test blocksize(b) == (2, 2)
    @test storedlength(b) == 1 * 1 + 2 * 2
    @test blockstoredlength(b) == 2
    for f in (getindex, view)
      # TODO: Broken on GPU.
      @allowscalar begin
        @test size(f(b, Block(1, 1))) == (1, 2)
        @test size(f(b, Block(2, 1))) == (2, 2)
        @test size(f(b, Block(1, 2))) == (1, 1)
        @test size(f(b, Block(2, 2))) == (2, 1)
        @test f(b, Block(1, 1)) == a[Block(1, 1)[2:2, 2:3]]
        @test f(b, Block(2, 1)) == a[Block(2, 1)[1:2, 2:3]]
        @test f(b, Block(1, 2)) == a[Block(1, 2)[2:2, 1:1]]
        @test f(b, Block(2, 2)) == a[Block(2, 2)[1:2, 1:1]]
      end
    end
  end

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = a[Block(2, 1)[1:2, 2:3]]
  @test b == Array(a)[3:4, 2:3]
  @test size(b) == (2, 2)
  @test blocksize(b) == (1, 1)
  @test storedlength(b) == 2 * 2
  @test blockstoredlength(b) == 1

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = PermutedDimsArray(a, (2, 1))
  @test blockstoredlength(b) == 2
  @test Array(b) == permutedims(Array(a), (2, 1))
  c = 2 * b
  @test blockstoredlength(c) == 2
  @test Array(c) == 2 * permutedims(Array(a), (2, 1))

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = a'
  @test blockstoredlength(b) == 2
  @test Array(b) == Array(a)'
  c = 2 * b
  @test blockstoredlength(c) == 2
  @test Array(c) == 2 * Array(a)'

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = transpose(a)
  @test blockstoredlength(b) == 2
  @test Array(b) == transpose(Array(a))
  c = 2 * b
  @test blockstoredlength(c) == 2
  @test Array(c) == 2 * transpose(Array(a))

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = a[Block(1), Block(1):Block(2)]
  @test size(b) == (2, 7)
  @test blocksize(b) == (1, 2)
  @test b[Block(1, 1)] == a[Block(1, 1)]
  @test b[Block(1, 2)] == a[Block(1, 2)]

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = copy(a)
  x = randn(elt, size(@view(a[Block(2, 2)])))
  b[Block(2), Block(2)] = x
  @test b[Block(2, 2)] == x

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = copy(a)
  b[Block(1, 1)] .= 1
  @test b[Block(1, 1)] == trues(blocksizes(b)[1, 1])

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  b = @view a[Block(2, 2)]
  @test size(b) == (3, 4)
  for i in parentindices(b)
    @test i isa Base.OneTo{Int}
  end
  @test parentindices(b)[1] == 1:3
  @test parentindices(b)[2] == 1:4

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  b = @view a[Block(2, 2)[1:2, 2:2]]
  @test size(b) == (2, 1)
  for i in parentindices(b)
    @test i isa UnitRange{Int}
  end
  @test parentindices(b)[1] == 1:2
  @test parentindices(b)[2] == 2:2

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  x = randn(elt, 1, 2)
  @view(a[Block(2, 2)])[1:1, 1:2] = x
  @test a[Block(2, 2)][1:1, 1:2] == x
  @test @view(a[Block(2, 2)])[1:1, 1:2] == x
  @test a[3:3, 4:5] == x

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  x = randn(elt, 1, 2)
  @views a[Block(2, 2)][1:1, 1:2] = x
  @test a[Block(2, 2)][1:1, 1:2] == x
  @test @view(a[Block(2, 2)])[1:1, 1:2] == x
  @test a[3:3, 4:5] == x

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  b = @views a[Block(2, 2)][1:2, 2:3]
  @test b isa SubArray{<:Any,<:Any,<:BlockView}
  for i in parentindices(b)
    @test i isa UnitRange{Int}
  end
  x = randn(elt, 2, 2)
  b .= x
  @test a[Block(2, 2)[1:2, 2:3]] == x
  @test a[Block(2, 2)[1:2, 2:3]] == b
  @test blockstoredlength(a) == 1

  # Noncontiguous slicing.
  a = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
  a[Block(1, 1)] = dev(randn(elt, 2, 2))
  a[Block(2, 2)] = dev(randn(elt, 3, 3))
  I = ([3, 5], [2, 4])
  @test Array(a[I...]) == Array(a)[I...]

  # Noncontiguous slicing.
  a = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
  a[Block(1, 1)] = dev(randn(elt, 2, 2))
  a[Block(2, 2)] = dev(randn(elt, 3, 3))
  I = (:, [2, 4])
  if arrayt === Array
    @test Array(a[I...]) == Array(a)[I...]
  else
    # TODO: Broken on GPU, fix this.
    @test_broken a[I...]
  end

  a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
  @views for b in [Block(1, 1), Block(2, 2)]
    a[b] = randn(elt, size(a[b]))
  end
  for I in (Block.(1:2), [Block(1), Block(2)])
    b = @view a[I, I]
    for I in CartesianIndices(a)
      @test b[I] == a[I]
    end
    for block in BlockRange(a)
      @test b[block] == a[block]
    end
  end

  a = dev(BlockSparseArray{elt}(undef, [2, 3], [2, 3]))
  @views for b in [Block(1, 1), Block(2, 2)]
    # TODO: Use `blocksizes(a)[Int.(Tuple(b))...]` once available.
    a[b] = dev(randn(elt, size(a[b])))
  end
  for I in ([Block(2), Block(1)],)
    b = @view a[I, I]
    @test b[Block(1, 1)] == a[Block(2, 2)]
    @test b[Block(2, 1)] == a[Block(1, 2)]
    @test b[Block(1, 2)] == a[Block(2, 1)]
    @test b[Block(2, 2)] == a[Block(1, 1)]
    @allowscalar begin
      @test b[1, 1] == a[3, 3]
      @test b[4, 4] == a[1, 1]
      b[4, 4] = 44
      @test b[4, 4] == 44
    end
  end

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  b = a[Block(2):Block(2), Block(1):Block(2)]
  @test blockstoredlength(b) == 1
  @test b == Array(a)[3:5, 1:end]

  a = BlockSparseArray{elt}(undef, ([2, 3, 4], [2, 3, 4]))
  # TODO: Define `block_diagindices`.
  @views for b in [Block(1, 1), Block(2, 2), Block(3, 3)]
    a[b] = randn(elt, size(a[b]))
  end
  for (I1, I2) in (
    (mortar([Block(2)[2:3], Block(3)[1:3]]), mortar([Block(2)[2:3], Block(3)[2:3]])),
    ([Block(2)[2:3], Block(3)[1:3]], [Block(2)[2:3], Block(3)[2:3]]),
  )
    for b in (a[I1, I2], @view(a[I1, I2]))
      # TODO: Rename `blockstoredlength`.
      @test blockstoredlength(b) == 2
      @test b[Block(1, 1)] == a[Block(2, 2)[2:3, 2:3]]
      @test b[Block(2, 2)] == a[Block(3, 3)[1:3, 2:3]]
    end
  end

  a = dev(BlockSparseArray{elt}(undef, ([3, 3], [3, 3])))
  # TODO: Define `block_diagindices`.
  @views for b in [Block(1, 1), Block(2, 2)]
    a[b] = dev(randn(elt, size(a[b])))
  end
  I = mortar([Block(1)[1:2], Block(2)[1:2]])
  b = a[:, I]
  @test b[Block(1, 1)] == a[Block(1, 1)][:, 1:2]
  @test b[Block(2, 1)] == a[Block(2, 1)][:, 1:2]
  @test b[Block(1, 2)] == a[Block(1, 2)][:, 1:2]
  @test b[Block(2, 2)] == a[Block(2, 2)][:, 1:2]
  @test blocklengths.(axes(b)) == ([3, 3], [2, 2])
  # TODO: Rename `blockstoredlength`.
  @test blocksize(b) == (2, 2)
  @test blockstoredlength(b) == 2

  a = BlockSparseArray{elt}(undef, ([2, 3], [3, 4]))
  @views for b in [Block(1, 2), Block(2, 1)]
    a[b] = randn(elt, size(a[b]))
  end
  @test isassigned(a, 1, 1)
  @test isassigned(a, 1, 1, 1)
  @test !isassigned(a, 1, 1, 2)
  @test isassigned(a, 5, 7)
  @test isassigned(a, 5, 7, 1)
  @test !isassigned(a, 5, 7, 2)
  @test !isassigned(a, 0, 1)
  @test !isassigned(a, 5, 8)
  @test isassigned(a, Block(1), Block(1))
  @test isassigned(a, Block(2), Block(2))
  @test !isassigned(a, Block(1), Block(0))
  @test !isassigned(a, Block(3), Block(2))
  @test isassigned(a, Block(1, 1))
  @test isassigned(a, Block(2, 2))
  @test !isassigned(a, Block(1, 0))
  @test !isassigned(a, Block(3, 2))
  @test isassigned(a, Block(1)[1], Block(1)[1])
  @test isassigned(a, Block(2)[3], Block(2)[4])
  @test !isassigned(a, Block(1)[0], Block(1)[1])
  @test !isassigned(a, Block(2)[3], Block(2)[5])
  @test !isassigned(a, Block(1)[1], Block(0)[1])
  @test !isassigned(a, Block(3)[3], Block(2)[4])

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  @test iszero(a)
  @test iszero(blockstoredlength(a))
  fill!(a, 0)
  @test iszero(a)
  @test iszero(blockstoredlength(a))
  fill!(a, 2)
  @test !iszero(a)
  @test all(==(2), a)
  @test blockstoredlength(a) == 4
  fill!(a, 0)
  @test iszero(a)
  @test iszero(blockstoredlength(a))

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  @test iszero(a)
  @test iszero(blockstoredlength(a))
  zero!(a)
  @test iszero(a)
  @test iszero(blockstoredlength(a))
  fill!(a, 2)
  @test !iszero(a)
  @test all(==(2), a)
  @test blockstoredlength(a) == 4
  zero!(a)
  @test iszero(a)
  @test iszero(blockstoredlength(a))

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  @test iszero(a)
  @test iszero(blockstoredlength(a))
  a .= 0
  @test iszero(a)
  @test iszero(blockstoredlength(a))
  a .= 2
  @test !iszero(a)
  @test all(==(2), a)
  @test blockstoredlength(a) == 4
  a .= 0
  @test iszero(a)
  @test iszero(blockstoredlength(a))

  # TODO: Broken on GPU.
  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  for I in (Block.(1:2), [Block(1), Block(2)])
    b = @view a[I, I]
    x = randn(elt, 3, 4)
    b[Block(2, 2)] = x
    # These outputs a block of zeros,
    # for some reason the block
    # is not getting set.
    # I think the issue is that:
    # ```julia
    # @view(@view(a[I, I]))[Block(1, 1)]
    # ```
    # creates a doubly-wrapped SubArray
    # instead of flattening down to a
    # single SubArray wrapper.
    @test a[Block(2, 2)] == x
    @test b[Block(2, 2)] == x
  end

  function f1()
    a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
    b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
    x = randn(elt, 3, 4)
    b[Block(1, 1)] .= x
    return (; a, b, x)
  end
  function f2()
    a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
    b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
    x = randn(elt, 3, 4)
    b[Block(1, 1)] = x
    return (; a, b, x)
  end
  for abx in (f1(), f2())
    (; a, b, x) = abx
    @test b isa SubArray{<:Any,<:Any,<:BlockSparseArray}
    @test blockstoredlength(b) == 1
    @test b[Block(1, 1)] == x
    @test @view(b[Block(1, 1)]) isa Matrix{elt}
    for blck in [Block(2, 1), Block(1, 2), Block(2, 2)]
      @test iszero(b[blck])
    end
    @test blockstoredlength(a) == 1
    @test a[Block(2, 2)] == x
    for blck in [Block(1, 1), Block(2, 1), Block(1, 2)]
      @test iszero(a[blck])
    end
    @test_throws DimensionMismatch b[Block(1, 1)] .= randn(2, 3)
  end

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  b = @views a[[Block(2), Block(1)], [Block(2), Block(1)]][Block(2, 1)]
  @test iszero(b)
  @test size(b) == (2, 4)
  x = randn(elt, 2, 4)
  b .= x
  @test b == x
  @test a[Block(1, 2)] == x
  @test blockstoredlength(a) == 1

  a = BlockSparseArray{elt}(undef, [4, 3, 2], [4, 3, 2])
  @views for B in [Block(1, 1), Block(2, 2), Block(3, 3)]
    a[B] = randn(elt, size(a[B]))
  end
  b = @view a[[Block(3), Block(2), Block(1)], [Block(3), Block(2), Block(1)]]
  @test b isa SubArray{<:Any,<:Any,<:BlockSparseArray}
  c = @view b[4:8, 4:8]
  @test c isa SubArray{<:Any,<:Any,<:BlockSparseArray}
  @test size(c) == (5, 5)
  @test blockstoredlength(c) == 2
  @test blocksize(c) == (2, 2)
  @test blocklengths.(axes(c)) == ([2, 3], [2, 3])
  @test size(c[Block(1, 1)]) == (2, 2)
  @test c[Block(1, 1)] == a[Block(2, 2)[2:3, 2:3]]
  @test size(c[Block(2, 2)]) == (3, 3)
  @test c[Block(2, 2)] == a[Block(1, 1)[1:3, 1:3]]
  @test size(c[Block(2, 1)]) == (3, 2)
  @test iszero(c[Block(2, 1)])
  @test size(c[Block(1, 2)]) == (2, 3)
  @test iszero(c[Block(1, 2)])

  x = randn(elt, 3, 3)
  c[Block(2, 2)] = x
  @test c[Block(2, 2)] == x
  @test a[Block(1, 1)[1:3, 1:3]] == x

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  b = @view a[[Block(2), Block(1)], [Block(2), Block(1)]]
  for index in parentindices(@view(b[Block(1, 1)]))
    @test index isa Base.OneTo{Int}
  end

  a = BlockSparseArray{elt}(undef, [2, 3], [3, 4])
  a[Block(1, 1)] = randn(elt, 2, 3)
  b = @view a[Block(1, 1)[1:2, 1:1]]
  @test b isa SubArray{elt,2,Matrix{elt}}
  for i in parentindices(b)
    @test i isa UnitRange{Int}
  end

  a = BlockSparseArray{elt}(undef, [2, 2, 2, 2], [2, 2, 2, 2])
  @views for I in [Block(1, 1), Block(2, 2), Block(3, 3), Block(4, 4)]
    a[I] = randn(elt, size(a[I]))
  end
  for I in (blockedrange([4, 4]), BlockedVector(Block.(1:4), [2, 2]))
    b = @view a[I, I]
    @test copy(b) == a
    @test blocksize(b) == (2, 2)
    @test blocklengths.(axes(b)) == ([4, 4], [4, 4])
    # TODO: Fix in Julia 1.11 (https://github.com/ITensor/ITensors.jl/pull/1539).
    if VERSION < v"1.11-"
      @test b[Block(1, 1)] == a[Block.(1:2), Block.(1:2)]
      @test b[Block(2, 1)] == a[Block.(3:4), Block.(1:2)]
      @test b[Block(1, 2)] == a[Block.(1:2), Block.(3:4)]
      @test b[Block(2, 2)] == a[Block.(3:4), Block.(3:4)]
    end
    c = @view b[Block(2, 2)]
    @test blocksize(c) == (1, 1)
    @test c == a[Block.(3:4), Block.(3:4)]
  end

  a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = randn(elt, 2, 2)
  a[Block(2, 2)] = randn(elt, 3, 3)
  for I in (mortar([Block(1)[2:2], Block(2)[2:3]]), [Block(1)[2:2], Block(2)[2:3]])
    b = @view a[:, I]
    @test b == Array(a)[:, [2, 4, 5]]
  end

  # Merge and permute blocks.
  a = BlockSparseArray{elt}(undef, [2, 2, 2, 2], [2, 2, 2, 2])
  @views for I in [Block(1, 1), Block(2, 2), Block(3, 3), Block(4, 4)]
    a[I] = randn(elt, size(a[I]))
  end
  for I in (
    BlockVector([Block(4), Block(3), Block(2), Block(1)], [2, 2]),
    BlockedVector([Block(4), Block(3), Block(2), Block(1)], [2, 2]),
  )
    b = @view a[I, I]
    J = [Block(4), Block(3), Block(2), Block(1)]
    @test b == a[J, J]
    @test copy(b) == a[J, J]
    @test blocksize(b) == (2, 2)
    @test blocklengths.(axes(b)) == ([4, 4], [4, 4])
    @test b[Block(1, 1)] == Array(a)[[7, 8, 5, 6], [7, 8, 5, 6]]
    c = @views b[Block(1, 1)][2:3, 2:3]
    @test c == Array(a)[[8, 5], [8, 5]]
    @test copy(c) == Array(a)[[8, 5], [8, 5]]
    c = @view b[Block(1, 1)[2:3, 2:3]]
    @test c == Array(a)[[8, 5], [8, 5]]
    @test copy(c) == Array(a)[[8, 5], [8, 5]]
  end

  # TODO: Add more tests of this, it may
  # only be working accidentally.
  a = BlockSparseArray{elt}(undef, [2, 3], [2, 3])
  a[Block(1, 1)] = randn(elt, 2, 2)
  a[Block(2, 2)] = randn(elt, 3, 3)
  @test a[2:4, 4] == Array(a)[2:4, 4]
  # TODO: Fix this.
  @test_broken a[4, 2:4] == Array(a)[4, 2:4]
end
