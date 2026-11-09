# moongrad

A reverse-mode automatic differentiation engine, a tensor library, and a small
deep-learning stack — written from scratch in MoonBit.

`moongrad` compiles a program into a computation graph, then compiles the
*derivative* of that program into a second graph. The gradient is therefore a
first-class value: you can print it, count its nodes, re-evaluate it at a
different input, or differentiate it again to get a Hessian.

```moonbit
import { "logicore-code/moongrad" }

fn main {
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([0.7, -1.3]))

  let x = g.slice(g.variable(0), 0, 0, 1)
  let y = g.slice(g.variable(0), 0, 1, 1)
  let f = g.add(g.mul(g.pow(x, 2.0), y), g.exp(x))

  let grads = @moongrad.t_grad(g, b, f)
  println(grads.input_grads[0].to_data())   // [0.1938, 0.49]
}
```

## Why the gradient is a program

Most autodiff engines accumulate numeric adjoints into a tape: the forward pass
stores every intermediate value, and the backward pass reads them back and
pushes numbers into an array. That works, and it has two consequences worth
naming.

The first is that the gradient is *frozen at the point it was computed*. A
tape-based engine that asks for the gradient again after moving the weights
hands back the answer for the weights it used to have.

The second is that second-order support needs a second engine. Differentiating
a gradient means differentiating numbers that were already collapsed, which
loses the structure you would need.

Emitting a program avoids both. The backward sweep walks the graph backwards
and, wherever a node has a nonzero adjoint, emits the arithmetic for
`adjoint * local derivative` into the graph. The result is a program that can
be evaluated anywhere.

Two things follow directly:

- **Training compiles once.** `train_tensor.mbt` builds the graph once and
  compiles the gradient once; every step re-evaluates the compiled gradient at
  the new weights.
- **The Hessian is nearly free.** `H[i][j]` is just another call to the same
  compiler with the first sweep's output as the target, because that output is
  a node and not a number.

## What is in the box

| Module | What it holds |
|---|---|
| `tensor.mbt` | The tensor type: shape, row-major buffer, broadcasting, indexing, reshape, transpose, concat, slice |
| `tensor_op.mbt` | Elementwise arithmetic with broadcasting, matrix product, reductions, softmax, log-softmax, cross-entropy |
| `tensor_grad.mbt` | The tensor gradient compiler |
| `nn_tensor.mbt` | Dense layers, embeddings, dropout, sequential containers |
| `conv.mbt` | 2-D convolution, with im2col forward and a col2im backward |
| `linalg.mbt` | Eigenvalues, Cholesky, linear solve, QR, singular values, inverse, determinant, matrix exponential |
| `optim_tensor.mbt` | SGD, momentum, Adam, with per-element state |
| `train_tensor.mbt` | The training loop |
| `node.mbt`, `graph.mbt`, `simplify.mbt`, `forward.mbt`, `reverse.mbt`, `hessian.mbt`, `checkpoint.mbt`, `optim.mbt`, `train.mbt` | The scalar engine, which the tensor layer builds on and which still carries the Hessian / Newton / checkpointing machinery |

The scalar engine is not legacy: it is the rank-0 case of the same ideas, and
it is where the second-order and memory work lives. The tensor layer is the
general case.

### The tensor subset

Shapes broadcast when, aligned from the right, every axis is either equal or
one of them is 1 — numpy's rule, so `(m, 1) + (n,)` does what everybody
expects without anybody writing it out.

The gradient compiler handles: elementwise unary and binary operators with
broadcasting, matrix product, reduction along an axis, reshape, transpose,
concatenation, slicing, power, convolution, gather, and scaling by a
constant.

### The layer set

`Linear`, `Conv2d`, `Embedding`, `Dropout`, `Sequential`, and the activations
`identity`, `tanh`, `sigmoid`, `relu`, `softplus`.

## Correctness

The gradients are checked against **central differences of the same
function** — an oracle that shares no code with the compiler and therefore
catches a wrong rule rather than merely a typo.

`examples/gradcheck` runs that comparison for six functions covering
polynomials, quotients, matrix products, broadcasting products, chains of
nonlinearities, and logarithms. All six agree to the last bit.

The test suite (202 cases) additionally checks each local derivative rule
against its closed form, and each layer against a hand-computed
backpropagation.

`examples/nn_classifier` trains a two-layer network on two clusters: the loss
falls from 0.74 to 0.0003 over 400 steps and all eight predictions land on
the right side.

## Examples

```bash
moon run examples/tensor_basics --target native   # shapes and broadcasting
moon run examples/gradcheck     --target native   # gradients vs finite differences
moon run examples/nn_classifier  --target native   # a network that trains
moon run examples/linalg_demo   --target native   # eigenvalues, Cholesky, QR
moon run examples/derivatives   --target native   # the scalar engine
moon run examples/hessian       --target native   # second-order derivatives
moon run examples/checkpoint    --target native   # the memory / arithmetic trade
moon run examples/polyfit       --target native   # regression with the scalar engine
moon run examples/xor           --target native   # a linear model against a two-layer one
moon run examples/cse           --target native   # what the graph compiler saves
```

## Building

```bash
moon test          # 202 unit tests
moon check         # 0 warnings, 0 errors
moon fmt           # format
moon build         # build for the preferred target (wasm)
```

The library targets `wasm`, `wasm-gc`, `js` and `native` from the same source
with no conditional compilation.

## The one real limitation

ReLU's derivative is discontinuous, so it cannot be expressed as a smooth
function of the operand, and the IR has no comparison with which to say "if
the runtime value is positive". The compiler therefore reads the primal
value and emits a literal 0 or 1.

That means **a ReLU's gradient is correct at the point it was compiled and
has to be recompiled for another point**. Everything else in the library is
re-evaluable, and the tests pin the behaviour on both sides of the kink so
the boundary is documented rather than accidental.

A system that wanted a re-evaluable ReLU gradient would need a conditional in
the IR, which means a forward pass with branches — a different design, not a
small patch.

## Documentation

- `docs/architecture.md` — the three decisions that shape everything else
- `docs/math.md` — the derivative rules the compiler emits, one operator at a time
- `docs/benchmark.md` — measured numbers and their caveats

## License

Apache-2.0. See `LICENSE`.
