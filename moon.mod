// Module configuration for moongrad, a reverse-mode automatic
// differentiation engine written in pure MoonBit.
//
// The idea the project is built around: the backward pass does not
// accumulate numbers, it emits a second program. The gradient is
// therefore a first-class value — printable, re-evaluable at a
// different input, and differentiable again to get a Hessian.
//
// The interpreter covers the parts of the Scheme-like subset that
// matter for numerical work: lexical scoping, closures, `let`,
// `let*`, `letrec`, named `let`, recursion, and a library of
// arithmetic and list primitives.
//
// Learn more at https://docs.moonbitlang.com/en/latest/toolchain/moon/module.html

name = "logicore-code/moongrad"

version = "0.2.0"

readme = "README.md"

repository = "https://github.com/logicore-code/moongrad"

license = "Apache-2.0"

keywords = [
  "automatic-differentiation",
  "gradient",
  "machine-learning",
  "compiler",
  "differentiable-programming",
  "wasm",
  "moonbit",
]

preferred_target = "wasm"

description = "Reverse-mode automatic differentiation for MoonBit. The forward pass is constant propagation over a hash-consed DAG; the backward pass is a symbolic compiler that emits a gradient program rather than accumulating adjoints. Includes common-subexpression elimination, a conservative peephole simplifier, forward-over-reverse Hessians, a damped Newton step, and segment-recomputation gradient checkpointing. 151 unit tests, zero compiler warnings, no dependencies beyond the MoonBit core prelude."
