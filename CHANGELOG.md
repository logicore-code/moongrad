# Changelog

All notable changes to `logicore-code/moongrad` are recorded here. The
project follows Semantic Versioning: bump the

- `MAJOR` part on incompatible API changes
- `MINOR` part on new functionality in a backward-compatible way
- `PATCH` part on backward-compatible bug fixes

## [0.5.0] — 2026-11-11

### Added
- **Mini-batch training** in `train_batch.mbt`. The batch is an input,
  so the graph is compiled once for the whole run and the same
  compiled gradient serves every batch. The dataset is shuffled
  deterministically and the epoch length accounts for a short final
  batch.
- **Hessian-vector products** in `optim_second.mbt`, over the
  parameter space and over the input space. The direction is an input
  to the compiled program rather than a constant inside it, so one
  compilation serves every direction.
- **Newton-CG** in `optim_second.mbt`: a Newton step computed by
  conjugate gradients on the damped Hessian system, with the Hessian
  never formed. A dense Newton step over the explicit Hessian is
  provided alongside it, and the two agree.
- **Krylov solvers** in `linalg_krylov.mbt`: conjugate gradient for
  symmetric positive definite systems, GMRES for general ones, the
  full Jacobi eigendecomposition with its eigenvectors, the symmetric
  part of a matrix, and a spectral-norm bound.
- **More optimisers** in `optim_extra.mbt`: RMSProp, Adagrad, Nesterov
  momentum and AdamW with decoupled weight decay.
- **Gradient clipping** by norm and by value, and **learning-rate
  schedules**: constant, step, exponential and cosine.
- **Normalisation layers** in `nn_norm.mbt`: layer normalisation and
  batch normalisation, written once and specialised by axis.
- **More activations**: the sigmoid linear unit and the Gaussian error
  linear unit, in both engines.
- **Mean reductions** over an axis, with and without keeping the axis.
- `examples/minibatch` and `examples/second_order`.

### Fixed
- The **tensor trainer recompiled the gradient at every step**, which
  contradicted the design it was demonstrating and cost about twenty
  times the work. The gradient program is now compiled once, when the
  trainer is built, and re-evaluated at each step.
- `t_sum_axis_keep` wrote its output with the input's strides rather
  than the reduced ones, so a reduction that kept an axis longer than
  one read out of bounds. The mini-batch work found it.
- A gradient compiled against an empty binding baked the wrong shapes
  into the program. The trainers now fill the binding before
  compiling, and a compiled gradient records the shapes it was
  compiled for and checks them at evaluation, so a mismatch is
  reported instead of returning the gradient of a different function.

### Changed
- The test suite grew from 212 cases to 299.

## [0.4.0] — 2026-11-08

### Added
- The tensor backward sweep emits a **gradient program** rather than
  accumulating numeric adjoints. Every adjoint is now a node in the
  graph, and every local derivative is an expression over the primal's
  own nodes, so the tensor gradient is re-evaluable at any binding.
- New tensor nodes: `Adjust` (broadcast up, sum down), `ExpandAxis`,
  `BroadcastTo`, `PadTo`, `CropTo`, `ReluDeriv`, `Im2Col`, `Col2Im`,
  `ScatterRow`, `Select` and `Scatter`. The last two make a single
  element of a gradient addressable, which is what the Hessian needs.
- `ReluDeriv` in both engines. ReLU's derivative is a step function, and
  folding it into a literal at compile time produced a gradient that was
  right at the compile point and wrong everywhere else — and wrong in a
  way that looked right. The comparison now lives in a node, so it is
  decided where the gradient is evaluated.
- `t_hessian`, `t_param_hessian` and `t_mixed_hessian` in
  `tensor_hessian.mbt`: the Hessian of a tensor objective over the
  flattened input space, the parameter-against-parameter block, and the
  parameter-against-input block a Gauss-Newton step needs. All three
  come from differentiating the gradient program, so no second-order
  case exists in the compiler.
- `examples/reuse`, which compiles one gradient and evaluates it at four
  inputs, two of them on the far side of a ReLU's kink.
- `examples/tensor_hessian`, which checks the tensor Hessian against the
  closed form and against a finite difference of the gradient, and then
  takes a Newton step with it.
- `Matrix::nrows` and `Matrix::ncols`.

### Fixed
- The scalar engine's **parameter gradients** were read out of a frozen
  copy of the primal's values, so a gradient program evaluated at a
  binding other than the one it was compiled against produced the
  gradient of the wrong function. The variable gradients were correct;
  the parameter gradients were one step behind. Training masked it,
  because the loop refreshed the primal before every step, but the API
  was a trap. The gradient program is now evaluated as a whole, so the
  primal is re-evaluated with it.
- `t_sum_to_shape` returns the operand's own shape rather than the
  broadcast shape, which had been leaving the gradient of a broadcast
  operand at the wrong size.

### Changed
- `compile_gradient` no longer takes the primal's values: every local
  derivative is an expression now, so the sweep reads no operand value
  at all.
- `param_gradients` and `flat_param_gradients` can raise `ADError`,
  because they evaluate the gradient program.
- `checkpoint.mbt` now segments the *evaluation* of the gradient
  program rather than the sweep. The sweep no longer holds a tape, so
  the memory that remains is the evaluation's, and that is the quantity
  the module reports.
- The test suite grew from 202 cases to 212, and now includes tests that
  pin the gradient's re-evaluation and the tensor Hessian.

## [0.3.0] — 2026-10-20

### Added
- A tensor library: shape, row-major buffer, numpy broadcasting,
  indexing, reshape, transpose, concatenation, slicing and gather.
- Tensor-level reverse-mode differentiation covering elementwise unary
  and binary operators with broadcasting, matrix product, reduction
  along an axis, reshape, transpose, concatenation, slicing, power,
  convolution, gather and scaling by a constant.
- A neural-network layer set: dense layers, 2-D convolution, embeddings,
  dropout and sequential containers, with the activations identity,
  tanh, sigmoid, ReLU and softplus.
- Dense linear algebra: Jacobi eigenvalues, Cholesky, symmetric and
  general linear solves, QR by Householder reflections, singular values,
  inverse, determinant and matrix exponential.
- Optimisers with per-element state: SGD, momentum and Adam with the
  bias correction.
- A training loop that compiles the graph and the gradient once and
  re-evaluates them at every step.
- Ten runnable examples, including a gradient check against central
  differences and a network that trains.

### Fixed
- Four bugs that made the neural network silently not train: the
  parameter gradient was not reaching the optimiser, the bias term was
  not being updated, the learning rate was applied twice, and the loss
  was being read from the wrong slot.

## [0.2.0] — 2026-10-13

### Added
- Segment-recomputation gradient checkpointing in `checkpoint.mbt`:
  peak occupancy drops from O(n) to O(k + w) for k segments of
  width w, at the cost of re-evaluating every interior node once.
  The gradient is bit-identical in every configuration.
- Second-order derivatives in `hessian.mbt`, together with a
  Cholesky solver, a damped Newton step, and a Jacobi eigenvalue
  routine.
- A neural-network layer in `nn.mbt`: dense layers, tanh / sigmoid /
  ReLU / identity activations, MSE and numerically-stable binary
  cross-entropy, and a deterministic LCG so every example gives the
  same answer on every machine.
- Optimisers in `optim.mbt`: SGD, momentum, and Adam with the bias
  correction and the epsilon spelled out.
- A training loop in `train.mbt` that compiles the graph and the
  gradient once and re-evaluates them every step.
- A row-major matrix type in `tensor.mbt`.
- Six runnable examples under `examples/`: a derivative table
  checked against finite differences, polynomial regression, XOR, a
  Hessian walkthrough, the checkpointing trade-off, and what the
  graph compiler saves.
- 62 more unit tests: 89 to 151.

### Fixed
- The simplifier is literal-aware: `x - x` folds to zero instead of
  being kept as a subtraction, and a constant expression is folded once
  rather than re-derived on every use.
- Parameter gradients are accumulated as circuits rather than
  numbers, so they are re-evaluable instead of frozen at the point
  the graph was compiled.
- `Graph::variable` records the slot it interned a variable into.
  Before, a variable the caller declared but never used got a slot
  appended above the region the reverse sweep was walking, and the
  sweep read past the end of its own adjoint array.
- `cholesky_solve` appended back-substitution results in reverse
  index order, because the elimination runs right to left and the
  solver pushed instead of assigning.
- The trainer copies the optimiser's flat parameter array back into
  the parameter matrices after each step. Without it the parameters
  kept reporting their initial values while the training run
  converged on the new ones.

### Changed
- `moon check` reports 0 warnings and 0 errors, down from 19
  warnings. Most of that was removing `raise` from the signatures of
  functions that cannot fail.
- `Param::set_grad` became `Param::set_values`: it was writing the
  parameter values back after an optimiser step, never a gradient.

## [0.1.0] — 2026-10-13

The first release. The gradient compiler:

- `node.mbt` — the IR, with the index-ordering invariant that lets
  both passes be plain scans.
- `graph.mbt` — the builder, with hash-consing so common
  subexpressions are shared without a separate pass.
- `simplify.mbt` — a deliberately conservative peephole pass, and
  the local derivative tables.
- `forward.mbt` — constant propagation, plus an independent
  forward-mode dual-number engine used as a test oracle.
- `reverse.mbt` — the gradient compiler.
- `error.mbt` — `ADError` as a suberror, so a failure comes back to
  the caller with its operands attached instead of taking the process
  down.
- 89 unit tests covering the IR, the simplifier, both passes and
  the gradient compiler.

Apache-2.0 licensed throughout.
