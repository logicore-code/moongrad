name = "logicore-code/moongrad"

version = "0.1.0"

readme = "README.md"

repository = "https://github.com/logicore-code/moongrad"

license = "Apache-2.0"

keywords = [
  "automatic-differentiation",
  "machine-learning",
  "compiler",
  "differentiable-programming",
  "wasm",
  "moonbit",
]

preferred_target = "wasm"

description = "Reverse-mode automatic differentiation for MoonBit. Forward pass is constant propagation over a hash-consed DAG; the backward pass is a symbolic compiler that emits a gradient program. Includes common-subexpression elimination, algebraic simplification, forward-over-reverse Hessians, and segment-recomputation gradient checkpointing."
