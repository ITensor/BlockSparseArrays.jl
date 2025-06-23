using BlockArrays:
  Block,
  BlockIndex,
  BlockSlice,
  BlockedArray,
  BlockedVector,
  block,
  blockedrange,
  blockindex,
  mortar
using BlockSparseArrays:
  BlockSparseArrays,
  BlockIndexVector,
  BlockIndices,
  GenericBlockIndex,
  blocksparsezeros,
  blockedunitrange_getindices,
  to_block,
  to_block_indices,
  to_blockindexrange
using Test: @test, @test_broken, @testset

# blockrange
# checkindex
# to_indices
# to_index
# blockedunitrange_getindices
# viewblock
# to_blockindexrange

@testset "GenericBlockIndex" begin
  i1 = GenericBlockIndex(Block(1), ("x",))
  i2 = GenericBlockIndex(Block(2), ("y",))
  i = GenericBlockIndex(Block(1, 2), ("x", "y"))
  @test sprint(show, i) == "Block(1, 2)[x, y]"
  @test i isa GenericBlockIndex{2,Tuple{Int64,Int64},Tuple{String,String}}
  @test GenericBlockIndex(Block(1), "x") === i1
  @test GenericBlockIndex(1, "x") === i1
  @test GenericBlockIndex(1, ("x",)) === i1
  @test GenericBlockIndex((1,), "x") === i1
  @test GenericBlockIndex((1, 2), ("x", "y")) === i
  @test GenericBlockIndex((Block(1), Block(2)), ("x", "y")) === i
  @test GenericBlockIndex((i1, i2)) === i
  @test block(i1) == Block(1)
  @test block(i) == Block(1, 2)
  @test blockindex(i1) == "x"
  @test GenericBlockIndex((), ()) == GenericBlockIndex(Block(), ())
  @test GenericBlockIndex(Block(1, 2), ("x",)) == GenericBlockIndex(Block(1, 2), ("x", 1))

  i1 = GenericBlockIndex(Block(1), (1,))
  i2 = GenericBlockIndex(Block(2), (2,))
  i = GenericBlockIndex(Block(1, 2), (1, 2))
  v = BlockedVector(["a", "b", "c", "d"], [2, 2])
  @test v[i1] == "a"
  @test v[i2] == "d"

  a = collect(Iterators.product(v, v))
  @test a[i1, i1] == ("a", "a")
  @test a[i2, i1] == ("d", "a")
  @test a[i1, i2] == ("a", "d")
  @test a[i] == ("a", "d")
  @test a[i2, i2] == ("d", "d")

  I = BlockIndexVector(Block(1), [1, 2])
  @test eltype(I) === BlockIndex{1,Tuple{Int},Tuple{Int}}
  @test ndims(I) === 1
  @test length(I) === 2
  @test size(I) === (2,)
  @test I[1] === Block(1)[1]
  @test I[2] === Block(1)[2]
  @test block(I) === Block(1)
  @test Block(I) === Block(1)
  @test copy(I) == BlockIndexVector(Block(1), [1, 2])

  I = BlockIndexVector(Block(1, 2), ([1, 2], [3, 4]))
  @test eltype(I) === BlockIndex{2,Tuple{Int,Int},Tuple{Int,Int}}
  @test ndims(I) === 2
  @test length(I) === 4
  @test size(I) === (2, 2)
  @test I[1, 1] === Block(1, 2)[1, 3]
  @test I[2, 1] === Block(1, 2)[2, 3]
  @test I[1, 2] === Block(1, 2)[1, 4]
  @test I[2, 2] === Block(1, 2)[2, 4]
  @test block(I) === Block(1, 2)
  @test Block(I) === Block(1, 2)
  @test copy(I) == BlockIndexVector(Block(1, 2), ([1, 2], [3, 4]))

  I = BlockIndexVector(Block(1), ["x", "y"])
  @test eltype(I) === GenericBlockIndex{1,Tuple{Int},Tuple{String}}
  @test ndims(I) === 1
  @test length(I) === 2
  @test size(I) === (2,)
  @test I[1] === GenericBlockIndex(Block(1), "x")
  @test I[2] === GenericBlockIndex(Block(1), "y")
  @test block(I) === Block(1)
  @test Block(I) === Block(1)
  @test copy(I) == BlockIndexVector(Block(1), ["x", "y"])

  I = BlockIndexVector(Block(1, 2), (["x", "y"], ["z", "w"]))
  @test eltype(I) === GenericBlockIndex{2,Tuple{Int,Int},Tuple{String,String}}
  @test ndims(I) === 2
  @test length(I) === 4
  @test size(I) === (2, 2)
  @test I[1, 1] === GenericBlockIndex(Block(1, 2), ("x", "z"))
  @test I[2, 1] === GenericBlockIndex(Block(1, 2), ("y", "z"))
  @test I[1, 2] === GenericBlockIndex(Block(1, 2), ("x", "w"))
  @test I[2, 2] === GenericBlockIndex(Block(1, 2), ("y", "w"))
  @test block(I) === Block(1, 2)
  @test Block(I) === Block(1, 2)
  @test copy(I) == BlockIndexVector(Block(1, 2), (["x", "y"], ["z", "w"]))

  v = BlockedVector(["a", "b", "c", "d"], [2, 2])
  i = BlockIndexVector(Block(1), [2, 1])
  @test v[i] == ["b", "a"]
  i = BlockIndexVector(Block(2), [2, 1])
  @test v[i] == ["d", "c"]

  v = BlockedVector(["a", "b", "c", "d"], [2, 2])
  i = BlockIndexVector{1,GenericBlockIndex{1,Tuple{Int},Tuple{String}}}(Block(1), [2, 1])
  @test v[i] == ["b", "a"]
  i = BlockIndexVector(Block(2), [2, 1])
  @test v[i] == ["d", "c"]

  a = collect(Iterators.product(v, v))
  i1 = BlockIndexVector(Block(1), [2, 1])
  i2 = BlockIndexVector(Block(2), [1, 2])
  i = BlockIndexVector(Block(1, 2), ([2, 1], [1, 2]))
  @test a[i1, i1] == [("b", "b") ("b", "a"); ("a", "b") ("a", "a")]
  @test a[i2, i1] == [("c", "b") ("c", "a"); ("d", "b") ("d", "a")]
  @test a[i1, i2] == [("b", "c") ("b", "d"); ("a", "c") ("a", "d")]
  @test a[i] == [("b", "c") ("b", "d"); ("a", "c") ("a", "d")]
  @test a[i2, i2] == [("c", "c") ("c", "d"); ("d", "c") ("d", "d")]

  a = collect(Iterators.product(v, v))
  i1 = BlockIndexVector{1,GenericBlockIndex{1,Tuple{Int},Tuple{String}}}(Block(1), [2, 1])
  i2 = BlockIndexVector{1,GenericBlockIndex{1,Tuple{Int},Tuple{String}}}(Block(2), [1, 2])
  i = BlockIndexVector{2,GenericBlockIndex{2,Tuple{Int,Int},Tuple{String,String}}}(
    Block(1, 2), ([2, 1], [1, 2])
  )
  @test a[i1, i1] == [("b", "b") ("b", "a"); ("a", "b") ("a", "a")]
  @test a[i2, i1] == [("c", "b") ("c", "a"); ("d", "b") ("d", "a")]
  @test a[i1, i2] == [("b", "c") ("b", "d"); ("a", "c") ("a", "d")]
  @test a[i] == [("b", "c") ("b", "d"); ("a", "c") ("a", "d")]
  @test a[i2, i2] == [("c", "c") ("c", "d"); ("d", "c") ("d", "d")]

  r = blockedrange([2, 3])
  i = mortar([BlockIndexVector(Block(2), [1]), BlockIndexVector(Block(1), [1, 2])])
  # TODO: Check the indices make sense and are in bounds.
  @test BlockSparseArrays.blockrange(r, i) == [Block(2), Block(1)]

  r = blockedrange([2, 3])
  i = mortar([BlockIndexVector(Block(2), ["x"]), BlockIndexVector(Block(1), ["y", "z"])])
  # TODO: Check the indices make sense and are in bounds.
  @test BlockSparseArrays.blockrange(r, i) == [Block(2), Block(1)]

  i = GenericBlockIndex(Block(2), 1)
  # TODO: Is this a good definition?
  @test Base.to_index(i) === i

  r = blockedrange([2, 3])
  @test checkindex(Bool, r, GenericBlockIndex(Block(1), 1))
  @test checkindex(Bool, r, GenericBlockIndex(Block(1), 2))
  @test !checkindex(Bool, r, GenericBlockIndex(Block(1), 3))
  @test checkindex(Bool, r, GenericBlockIndex(Block(2), 1))
  @test checkindex(Bool, r, GenericBlockIndex(Block(2), 2))
  @test checkindex(Bool, r, GenericBlockIndex(Block(2), 3))
  @test !checkindex(Bool, r, GenericBlockIndex(Block(2), 4))
  @test !checkindex(Bool, r, GenericBlockIndex(Block(3), 1))

  a = BlockedArray(randn(5, 5), [2, 3], [2, 3])
  i = GenericBlockIndex(Block(1), 1)
  @test to_indices(a, (i, i)) == (1, 1)
  @test to_indices(a, axes(a), (i, i)) == (1, 1)
  i = GenericBlockIndex(Block(2), 2)
  @test to_indices(a, (i, i)) == (4, 4)
  @test to_indices(a, axes(a), (i, i)) == (4, 4)

  r = blockedrange([2, 3])
  i = mortar([BlockIndexVector(Block(2), [1, 3]), BlockIndexVector(Block(1), [2])])
  @test blockedunitrange_getindices(r, i) == mortar([[3, 5], [2]])

  r = blockedrange([2, 3])
  T = GenericBlockIndex{1,Tuple{Int},Tuple{Int}}
  i = mortar([
    BlockIndexVector{1,T}(Block(2), [1, 3]), BlockIndexVector{1,T}(Block(1), [2])
  ])
  @test blockedunitrange_getindices(r, i) == mortar([[3, 5], [2]])

  # Internal functions.
  @test to_block(Block(2)) === Block(2)
  @test to_block(Block(2)[2:3]) === Block(2)
  @test to_block(BlockIndexVector(Block(2), [1, 3])) === Block(2)
  @test to_block_indices(Block(2)) === (:)
  @test to_block_indices(Block(2)[2:3]) === 2:3
  @test to_block_indices(BlockIndexVector(Block(2), [1, 3])) == [1, 3]

  a = blocksparsezeros([2, 3], [2, 3])
  i = Block(2)[2:3]
  @test to_indices(a, (i, i)) ===
    to_indices(a, axes(a), (i, i)) ===
    (BlockSlice(i, 4:5), BlockSlice(i, 4:5))

  a = blocksparsezeros([2, 3], [2, 3])
  i = mortar([Block(2)[2:3], Block(1)[2:2]])
  for I in (to_indices(a, (i, i)), to_indices(a, axes(a), (i, i)))
    for Iⱼ in I
      @test Iⱼ == BlockIndices(i, mortar([4:5, 2:2]))
      @test Iⱼ isa BlockIndices
      @test Iⱼ.blocks == i
      @test Iⱼ.indices == mortar([4:5, 2:2])
    end
  end

  a = blocksparsezeros([2, 3], [2, 3])
  i = BlockIndexVector(Block(2), [1, 3])
  for I in (to_indices(a, (i, i)), to_indices(a, axes(a), (i, i)))
    for Iⱼ in I
      @test Iⱼ == BlockIndices(i, [3, 5])
      @test Iⱼ isa BlockIndices
      @test Iⱼ.blocks == i
      @test Iⱼ.indices == [3, 5]
    end
  end

  a = blocksparsezeros([2, 3], [2, 3])
  T = GenericBlockIndex{1,Tuple{Int},Tuple{Int}}
  i = BlockIndexVector{1,T}(Block(2), [1, 3])
  for I in (to_indices(a, (i, i)), to_indices(a, axes(a), (i, i)))
    for Iⱼ in I
      @test Iⱼ == BlockIndices(i, [3, 5])
      @test Iⱼ isa BlockIndices
      @test Iⱼ.blocks == i
      @test Iⱼ.indices == [3, 5]
    end
  end

  a = blocksparsezeros([2, 3], [2, 3])
  i = mortar([BlockIndexVector(Block(2), [1, 3]), BlockIndexVector(Block(1), [2])])
  for I in (to_indices(a, (i, i)), to_indices(a, axes(a), (i, i)))
    for Iⱼ in I
      @test Iⱼ == BlockIndices(i, mortar([[3, 5], [2]]))
      @test Iⱼ isa BlockIndices
      @test Iⱼ.blocks == i
      @test Iⱼ.indices == mortar([[3, 5], [2]])
    end
  end

  a = blocksparsezeros([2, 3], [2, 3])
  i = [BlockIndexVector(Block(2), [1, 3]), BlockIndexVector(Block(1), [2])]
  for I in (to_indices(a, (i, i)), to_indices(a, axes(a), (i, i)))
    for Iⱼ in I
      @test Iⱼ == BlockIndices(mortar(i), mortar([[3, 5], [2]]))
      @test Iⱼ isa BlockIndices
      @test Iⱼ.blocks == mortar(i)
      @test Iⱼ.indices == mortar([[3, 5], [2]])
    end
  end

  a = blocksparsezeros([2, 3], [2, 3])
  T = GenericBlockIndex{1,Tuple{Int},Tuple{Int}}
  i = mortar([
    BlockIndexVector{1,T}(Block(2), [1, 3]), BlockIndexVector{1,T}(Block(1), [2])
  ])
  for I in (to_indices(a, (i, i)), to_indices(a, axes(a), (i, i)))
    for Iⱼ in I
      @test Iⱼ == BlockIndices(i, mortar([[3, 5], [2]]))
      @test Iⱼ isa BlockIndices
      @test Iⱼ.blocks == i
      @test Iⱼ.indices == mortar([[3, 5], [2]])
    end
  end

  a = blocksparsezeros([2, 3], [2, 3])
  T = GenericBlockIndex{1,Tuple{Int},Tuple{Int}}
  i = [BlockIndexVector{1,T}(Block(2), [1, 3]), BlockIndexVector{1,T}(Block(1), [2])]
  for I in (to_indices(a, (i, i)), to_indices(a, axes(a), (i, i)))
    for Iⱼ in I
      @test Iⱼ == BlockIndices(mortar(i), mortar([[3, 5], [2]]))
      @test Iⱼ isa BlockIndices
      @test Iⱼ.blocks == mortar(i)
      @test Iⱼ.indices == mortar([[3, 5], [2]])
    end
  end
end
