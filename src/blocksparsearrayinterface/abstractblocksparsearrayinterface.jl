using Derive: Derive, @interface
using SparseArraysBase: AbstractSparseArrayInterface

using SparseArraysBase: AbstractSparseArrayStyle

abstract type AbstractBlockSparseArrayStyle{N} <: AbstractSparseArrayStyle{N} end

struct BlockSparseArrayStyle{N} <: AbstractBlockSparseArrayStyle{N} end

# TODO: Add `ndims` type parameter.
# TODO: This isn't used to define interface functions right now.
# Currently, `@interface` expects an instance, probably it should take a
# type instead so fallback functions can use abstract types.
abstract type AbstractBlockSparseArrayInterface <: AbstractSparseArrayInterface end

