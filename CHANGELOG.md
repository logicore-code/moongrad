# Changelog

All notable changes to `logicore-code/moongrad` are recorded here. The
project follows Semantic Versioning: bump the

- `MAJOR` part on incompatible API changes
- `MINOR` part on new functionality in a backward-compatible way
- `PATCH` part on backward-compatible bug fixes

## [0.2.0] — 2026-10-13

### Added
- `letrec` for mutually-recursive local bindings, plus named `let`
  for the canonical `(let loop ((i 0) (acc 0)) ...)` iteration form.
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
- Plain-form `(define name expr)` is now recursive. Previously only
  the short-form `(define (name params...) body)` was, so a body
  referring to its own name raised `UnboundSymbol`.
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

The first release. The interpreter core:

- `node.mbt` — the IR, with the index-ordering invariant that lets
  both passes be plain scans.
- `graph.mbt` — the builder, with hash-consing so common
  subexpressions are shared without a separate pass.
- `simplify.mbt` — a deliberately conservative peephole pass, and
  the local derivative tables.
- `forward.mbt` — constant propagation, plus an independent
  forward-mode dual-number engine used as a test oracle.
- `reverse.mbt` — the gradient compiler.
- `error.mbt` — `ReadError` / `EvalError` / `ArityError` shaped
  suberrors, so a failure comes back to the caller with its
  operands attached instead of taking the process down.
- 89 unit tests covering the IR, the simplifier, both passes and
  the gradient compiler.

Apache-2.0 licensed throughout.
