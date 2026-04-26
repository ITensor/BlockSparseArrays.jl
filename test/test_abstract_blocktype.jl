using Adapt: adapt
using BlockArrays: Block
using BlockSparseArrays: BlockSparseMatrix, blockstoredlength
using JLArrays: JLArray
using LinearAlgebra: hermitianpart, norm
using MatrixAlgebraKit: eig_full, eig_trunc, eig_vals, eigh_full, eigh_trunc, eigh_vals,
    isisometric, left_orth, left_polar, lq_compact, lq_full, qr_compact, qr_full,
    right_orth, right_polar, svd_compact, svd_full, svd_trunc
using SparseArraysBase: storedlength
using StableRNGs: StableRNG
using Test: @test, @testset

elts = (Float32, Float64, ComplexF32)
arrayts = (Array, JLArray)
@testset "Abstract block type (arraytype=$arrayt, eltype=$elt)" for arrayt in arrayts,
        elt in elts

    dev = adapt(arrayt)

    a = BlockSparseMatrix{elt, AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
    @test sprint(show, MIME"text/plain"(), a) isa String
    @test iszero(storedlength(a))
    @test iszero(blockstoredlength(a))

    rng = StableRNG(1234)
    a = BlockSparseMatrix{elt, AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
    a[Block(1, 1)] = dev(randn(rng, elt, 2, 2))
    @test !iszero(a[Block(1, 1)])
    @test a[Block(1, 1)] isa arrayt{elt, 2}
    @test iszero(a[Block(2, 2)])
    @test a[Block(2, 2)] isa Matrix{elt}
    @test iszero(a[Block(2, 1)])
    @test a[Block(2, 1)] isa Matrix{elt}
    @test iszero(a[Block(1, 2)])
    @test a[Block(1, 2)] isa Matrix{elt}

    rng = StableRNG(1234)
    a = BlockSparseMatrix{elt, AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
    a[Block(1, 1)] = dev(randn(rng, elt, 2, 2))
    a′ = BlockSparseMatrix{elt, AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
    a′[Block(2, 2)] = dev(randn(rng, elt, 3, 3))

    b = copy(a)
    @test Array(b) ≈ Array(a)

    b = a + a′
    @test Array(b) ≈ Array(a) + Array(a′)

    b = 3a
    @test Array(b) ≈ 3Array(a)

    b = a * a
    @test Array(b) ≈ Array(a) * Array(a)

    b = a * a′
    @test Array(b) ≈ Array(a) * Array(a′)
    @test norm(b) ≈ 0

    rng = StableRNG(1234)
    a = BlockSparseMatrix{elt, AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
    a[Block(1, 1)] = dev(randn(rng, elt, 2, 2))
    for f in (eig_full, eig_trunc)
        @test begin
            d, v = f(a)
            a * v ≈ v * d
        end broken = arrayt ≢ Array
    end
    @test begin
        d = eig_vals(a)
        sort(Vector(d); by = abs) ≈ sort(eig_vals(Matrix(a)); by = abs)
    end broken = arrayt ≢ Array

    rng = StableRNG(1234)
    a = BlockSparseMatrix{elt, AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
    a[Block(1, 1)] = dev(parent(hermitianpart(randn(rng, elt, 2, 2))))
    for f in (eigh_full, eigh_trunc)
        @test begin
            d, v = f(a)
            a * v ≈ v * d
        end broken = arrayt ≢ Array
    end
    @test begin
        d = eigh_vals(a)
        sort(Vector(d); by = abs) ≈ sort(eig_vals(Matrix(a)); by = abs)
    end broken = arrayt ≢ Array

    rng = StableRNG(1234)
    a = BlockSparseMatrix{elt, AbstractMatrix{elt}}(undef, [2, 3], [2, 3])
    a[Block(1, 1)] = dev(randn(rng, elt, 2, 2))
    for f in (left_orth, left_polar, qr_compact, qr_full)
        u, c = f(a)
        @test u * c ≈ a
        # TODO: Fix comparison with UniformScaling on GPU.
        @test isisometric(u; side = :left) broken = arrayt ≢ Array
    end
    for f in (right_orth, right_polar, lq_compact, lq_full)
        c, u = f(a)
        # `right_orth`, `lq_compact`, `lq_full` on JLArray-backed
        # `BlockSparseMatrix{T, AbstractMatrix{T}}` produce a `c * u` that does not
        # reproduce `a` (regression in MatrixAlgebraKit 0.6.6, tracked at
        # https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/issues/218).
        @test c * u ≈ a broken = (arrayt ≢ Array && f ∈ (right_orth, lq_compact, lq_full))
        # TODO: Fix comparison with UniformScaling on GPU.
        @test isisometric(u; side = :right) broken = arrayt ≢ Array
    end
    for f in (svd_compact, svd_full, svd_trunc)
        # `svd_trunc` on JLArray-backed `BlockSparseMatrix{T, AbstractMatrix{T}}`
        # currently triggers scalar indexing on the GPU array.
        @test begin
            u, s, v = f(a)
            u * s * v ≈ a
        end broken = (arrayt ≢ Array && f ≡ svd_trunc)
    end
end
