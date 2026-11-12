# Examples

Fourteen executable packages, each demonstrating one slice of the engine.

```bash
moon run examples/tensor_basics  --target native
moon run examples/gradcheck       --target native
moon run examples/reuse           --target native
moon run examples/tensor_hessian  --target native
moon run examples/nn_classifier   --target native
moon run examples/linalg_demo     --target native
moon run examples/derivatives     --target native
moon run examples/hessian         --target native
moon run examples/checkpoint      --target native
moon run examples/polyfit         --target native
moon run examples/xor             --target native
moon run examples/cse             --target native
moon run examples/second_order    --target native
moon run examples/minibatch       --target native
```

| Example | What it shows | Headline output |
|---|---|---|
| `tensor_basics/` | shapes, broadcasting, matrix products, reductions | a `(2,3) + (3,)` broadcast to `(2,3)` |
| `gradcheck/` | six functions, analytic gradient vs central difference | they agree to the finite difference's own error |
| `reuse/` | one compiled gradient, evaluated at four different inputs | the gradient follows the input across a ReLU's kink |
| `tensor_hessian/` | the Hessian of a tensor objective | matches the closed form and a finite difference; one Newton step lands on the minimiser |
| `nn_classifier/` | a two-layer network trained by gradient descent | loss 0.92 → 0.0003, eight predictions right |
| `linalg_demo/` | eigenvalues, Cholesky, QR, singular values | a symmetric matrix's spectrum |
| `derivatives/` | seven functions, analytic gradient vs central difference | the two columns agree to four decimals |
| `hessian/` | the Hessian, its eigenvalues, a Newton step | one Newton step reaches the minimum |
| `checkpoint/` | the memory/arithmetic trade-off | less memory for more arithmetic |
| `polyfit/`     | recovering a quadratic's coefficients from noisy data | `1.5 -> 1.4954`, `-0.8 -> -0.8019` |
| `xor/`         | a linear model against a two-layer one | linear `0.25`, two-layer `0` |
| `cse/`         | what the graph compiler saves | slots, folds, identities |
| `second_order/` | the Hessian, a Hessian-vector product, Newton-CG | the dense and the matrix-free route agree |
| `minibatch/`   | mini-batch training on a synthetic dataset | the weight converges to the true one |

## Why `gradcheck` matters

Every analytic gradient is printed next to a central difference of the same
function. A gradient engine that is subtly wrong can still produce
plausible-looking numbers, so the check that matters is against an
independent method rather than against a table of expectations.

The same argument applies inside the test suite, where every non-trivial
gradient is checked against *two* independent derivations: the closed form
and the forward-mode dual-number engine in `forward.mbt`, which shares no
code with the reverse pass.

## Why `reuse` matters

It compiles a gradient once and then evaluates it at four different inputs.
Two of them are on the other side of a ReLU's kink, and the gradient
changes with the input rather than staying where it was compiled.

This is the check that a frozen gradient cannot pass. A backward pass that
folds ReLU's step function into a literal produces a number that is right
at the compile point and wrong everywhere else — and wrong in a way that
still looks like a plausible gradient, so nothing catches it by accident.

## Why `tensor_hessian` matters

The Hessian of a tensor objective is built by differentiating the gradient
program, which is possible only because the gradient is a program. The
example checks the result against the closed form *and* against a finite
difference of the gradient, then uses it to take a Newton step that lands
exactly on the minimiser of a quadratic.

A second-order engine that had to be written separately would have to
re-derive every rule a second time, and the two would disagree somewhere.

## Why `second_order` matters

It computes the same Newton step two ways: once from the dense
Hessian, with a Cholesky factorisation, and once without forming the
Hessian at all, using a Hessian-vector product inside a
conjugate-gradient iteration. The two agree exactly, which is what
says the matrix-free route is computing the right thing.

## Why `minibatch` matters

The batch is an input, so the graph is compiled once for the whole
run and the same compiled gradient serves every batch. The example
shows the loss falling over thirty epochs and the weight converging
to the one that generated the data.

## Why `checkpoint` matters

It prints what each segment count costs in memory and in arithmetic, and
asserts that the gradient is bit-identical in every configuration.

A memory strategy that quietly computes something slightly different is
worse than no memory strategy: the symptom is a training run that converges
to a marginally different minimum and nobody can say why.

## Why `xor` matters

It contrasts a linear model against a two-layer one on the same data. The
linear model converges to a loss of 0.25, which is the best it can do —
no line separates two diagonally opposite corners of a square from the other
two. The two-layer network reaches zero.

It is also where the sigmoid/logit pairing shows up: the output layer is
linear and the cross-entropy is computed from the logit, because feeding a
post-sigmoid probability into the loss squashes the gradient exactly where
it needs to be largest.
