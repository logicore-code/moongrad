// Module configuration for moongrad, a reverse-mode automatic
// differentiation engine written in pure MoonBit.
//
// The idea the project is built around: the backward pass does not
// accumulate numbers, it emits a second program. The gradient is
// therefore a first-class value — printable, re-evaluable at a
// different input, and differentiable again to get a Hessian.
//
// The library covers the numerical parts of machine learning: a
// multi-dimensional tensor type with numpy broadcasting, a tensor
// gradient compiler, dense and convolutional layers, normalisation,
// a choice of optimisers and learning-rate schedules, mini-batch
// training, dense linear algebra with Krylov solvers, and
// second-order optimisation through Hessian-vector products.
//
// Learn more at https://docs.moonbitlang.com/en/latest/toolchain/moon/module.html

name = "logicore-code/moongrad"

version = "0.6.0"

readme = "README.md"

repository = "https://github.com/logicore-code/moongrad"

license = "Apache-2.0"

keywords = [
  "automatic-differentiation",
  "gradient",
  "hessian",
  "machine-learning",
  "compiler",
  "differentiable-programming",
  "numerical-computing",
  "tensor",
  "wasm",
  "moonbit",
]

preferred_target = "wasm"

description = "Reverse-mode automatic differentiation for MoonBit. The forward pass is constant propagation over a hash-consed DAG; the backward pass is a symbolic compiler that emits a gradient program rather than accumulating adjoints. The program can be re-evaluated at any input and differentiated again, so the same machinery gives second derivatives and second-order optimisation. Includes multi-dimensional tensors with broadcasting, convolution, normalisation layers, mini-batch training, a choice of optimisers and schedules, dense linear algebra with Krylov solvers, and Hessian-vector products."
