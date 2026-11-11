# moongrad

A reverse-mode automatic differentiation engine, a tensor library, and a small
deep-learning stack — written from scratch in MoonBit.

`moongrad` compiles a program into a computation graph, then compiles the
*derivative* of that program into a second graph. The gradient is therefore a
first-class value: you can print it, count its nodes, re-evaluate it at a
different input, or differentiate it again to get a Hessian.

```moonbit
// main.mbt
fn main {
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([0.7, -1.3]))

  let x = g.slice(g.variable(0), 0, 0, 1)
  let y = g.slice(g.variable(0), 0, 1, 1)
  let f = g.add(g.mul(g.pow(x, 2.0), y), g.exp(x))

  let grads = @moongrad.t_grad(g, b, f)
  println(grads.input_grads[0].data)   // [0.1937, 0.4899]
}
```

```bash
# moon.pkg
import { "logicore-code/moongrad" }
pkgtype(kind: "executable")
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

That is the whole design, and it is what the rest of the library hangs off:

- **Training compiles once.** `train_tensor.mbt` builds the graph once and
  compiles the gradient once; every step re-evaluates the compiled gradient at
  the new weights.
- **The Hessian is nearly free.** `H[i][j]` is just another call to the same
  compiler with the first sweep's output as the target, because that output is
  a node and not a number.
- **A gradient is correct wherever it is evaluated.** Every local derivative
  is an expression over the primal's own nodes, so it is recomputed rather
  than remembered. This matters most for the operators whose derivative is not
  a smooth function — ReLU's is a step, and a step has to be evaluated where
  the input is known, not where the gradient was compiled.

## What is in the box

| Module | What it holds |
|---|---|
| `tensor.mbt` | The tensor type: shape, row-major buffer, broadcasting, indexing, reshape, transpose, concat, slice |
| `tensor_op.mbt` | Elementwise arithmetic with broadcasting, matrix product, reductions, softmax, log-softmax, cross-entropy |
| `tensor_grad.mbt` | The tensor gradient compiler: the backward sweep that emits a program |
| `tensor_shape.mbt` | Shape adjustments — cropping, padding, scattering a row back into a parameter |
| `tensor_hessian.mbt` | The Hessian of a tensor objective, and the mixed parameter-input block |
| `nn_tensor.mbt` | Dense layers, embeddings, dropout, sequential containers |
| `nn_norm.mbt` | Layer normalisation and batch normalisation, as one computation |
| `conv.mbt` | 2-D convolution, with im2col forward and a col2im backward |
| `linalg.mbt` | Eigenvalues, Cholesky, linear solve, QR, singular values, inverse, determinant, matrix exponential |
| `linalg_krylov.mbt` | Conjugate gradient, GMRES, the full Jacobi eigendecomposition, spectral norm |
| `optim_tensor.mbt` | SGD, momentum, Adam, with per-element state |
| `optim_extra.mbt` | RMSProp, Adagrad, Nesterov, AdamW; gradient clipping; learning-rate schedules |
| `optim_second.mbt` | Hessian-vector products, Newton-CG, the dense Newton step |
| `train_tensor.mbt` | The full-batch training loop |
| `train_batch.mbt` | The mini-batch training loop: shuffling, batching, epochs |
| `model_io.mbt` | Model parameters as JSON, with an exact round trip |
| `dataset.mbt` | Deterministic shuffling, batching, k-fold splits, synthetic data |
| `node.mbt`, `graph.mbt`, `simplify.mbt`, `forward.mbt`, `reverse.mbt`, `hessian.mbt`, `checkpoint.mbt`, `optim.mbt`, `train.mbt` | The scalar engine, which the tensor layer builds on and which still carries the Newton and checkpointing machinery |

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

It compiles them into a program rather than a number, so the gradient can be
evaluated at any binding — see `examples/reuse`. The second derivative comes
out of the same machinery: `t_hessian` differentiates the gradient program,
and `t_mixed_hessian` gives the parameter-against-input block that a
Gauss-Newton step needs.

### The layer set

`Linear`, `Conv2d`, `Embedding`, `Dropout`, `Sequential`, layer normalisation
and batch normalisation, and the activations `identity`, `tanh`, `sigmoid`,
`relu`, `softplus`, `silu`, `gelu`.

### Second-order optimisation

The engine can differentiate a gradient, so it can build a Hessian — and it can
also avoid building one. A Hessian-vector product is

```
H v = grad( g . v )
```

the gradient of the gradient dotted with the direction, which means the matrix
never appears. The direction is an *input* to the compiled program rather than a
constant inside it, so one compilation serves every direction and a
conjugate-gradient iteration can be run on the Hessian system with nothing
dense in sight.

That is Newton-CG. `examples/second_order` shows it against the dense route:
both compute the same Newton step, and the two agree exactly.

### Training

The graph is compiled once and the gradient is compiled once; every step
re-evaluates the compiled gradient at the new weights. `train_batch.mbt` takes
that further and puts the mini-batch in an input slot, so the same compiled
gradient serves every batch of the run.

## Correctness

The gradients are checked against **central differences of the same
function** — an oracle that shares no code with the compiler and therefore
catches a wrong rule rather than merely a typo.

`examples/gradcheck` runs that comparison for six functions covering
polynomials, quotients, matrix products, broadcasting products, chains of
nonlinearities, and logarithms. All six agree to the last bit.

The test suite (339 cases) additionally checks each local derivative rule
against its closed form, and each layer against a hand-computed
backpropagation.

The reusability claim has its own tests, because it is the kind of property
that can rot quietly. `tensor_program_test.mbt` compiles a gradient once and
then evaluates it at other inputs, including on the other side of a ReLU's
kink; a gradient frozen at its compile point fails those tests. The same file
checks the tensor Hessian against the closed form, against a finite difference
of the gradient, and for symmetry.

`examples/nn_classifier` trains a two-layer network on two clusters: the loss
falls from 0.92 to 0.0003 over 360 steps and all eight predictions land on
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
moon run examples/reuse         --target native   # one gradient, evaluated at many inputs
moon run examples/tensor_hessian --target native  # second derivatives on the tensor side
moon run examples/second_order  --target native   # Newton-CG, without forming the Hessian
moon run examples/minibatch     --target native   # mini-batch training
```

## Building

```bash
moon test          # 339 unit tests
moon check         # 0 warnings, 0 errors
moon fmt           # format
moon build         # build for the preferred target (wasm)
```

The library targets `wasm`, `wasm-gc`, `js` and `native` from the same source
with no conditional compilation.

## Where the gradient decides things

Most derivative rules are ordinary functions of the operand, and the compiler
emits them as expressions: `d/dx cos(x)` becomes `cos(x)` applied to the
primal's own node, `d/dx tanh(x)` becomes `1 - tanh(x)^2`, and so on. Those
re-evaluate themselves wherever the gradient is evaluated.

ReLU is the one operator whose derivative is not a function of the operand's
*value* in any smooth sense: it is one where the input is positive and zero
where it is negative. Folding that branch into a literal at compile time
gives a gradient that is right at the point it was compiled and wrong at every
other point — and wrong in a way that looks exactly like right, because the
number is still a plausible gradient.

So the compiler does not fold it. `ReluDeriv` is a node in the IR, and its
value is computed where the gradient is evaluated:

```
d/dx relu(x)  =  ReluDeriv(x)
ReluDeriv(x)  =  1  if x > 0
                 0  otherwise
```

The comparison lives in the node, not in the compiler, which is the difference
between a branch that is decided once and a branch that is decided every time
the gradient runs. `examples/reuse` compiles a gradient where the input is
positive and then evaluates it where it is negative; the gradient follows the
input.

The same node makes the second derivative of a ReLU zero almost everywhere,
which is the honest answer: the kink is the one place where the second
derivative does not exist.

## Design decisions

A few choices shape everything else here. They are decisions, with reasons,
rather than gaps to be filled in later.

**The graph is compiled for a set of shapes.** The backward rules need to know
each operand's shape — the shape is what says how a broadcast has to be summed
back down — so a program is valid for the shapes it was compiled against. This
is the same bargain every statically-shaped tensor library makes. Changing a
shape means recompiling, which is one call. To keep that from turning into a
silent wrong answer, a compiled gradient records the input shapes it was
compiled for and checks them at evaluation: a mismatch is reported rather than
returning the gradient of a different function.

**Intermediate tensors are not written in place.** In-place would save memory
and would break re-evaluation: the first evaluation would dirty the buffers
and the second would read corrupted values. Re-evaluability is the point of
the design, so the buffers stay and the memory question is answered by
`checkpoint.mbt` instead.

**Two engines, one derivative table.** The scalar engine is the rank-0 case of
the tensor one. They share the tables in `simplify.mbt`, so a rule fixed in
one is fixed in the other.

**Two routes to a second-order step.** The dense Hessian is exact and simple
when the parameter count is small. The matrix-free route — a
Hessian-vector product plus conjugate gradient — is what scales, and the two
agree to the last bit.

## Documentation

- `docs/architecture.md` — the three decisions that shape everything else
- `docs/math.md` — the derivative rules the compiler emits, one operator at a time
- `docs/benchmark.md` — measured numbers and their caveats

## License

Apache-2.0. See `LICENSE`.
