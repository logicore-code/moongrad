# Examples

Six executable packages, each demonstrating one slice of the engine.

```bash
moon run examples/derivatives --target native
moon run examples/polyfit     --target native
moon run examples/xor         --target native
moon run examples/hessian     --target native
moon run examples/checkpoint  --target native
moon run examples/cse         --target native
```

| Example | What it shows | Headline output |
|---|---|---|
| `derivatives/` | seven functions, analytic gradient vs central difference | the two columns agree to four decimals |
| `polyfit/`     | recovering a quadratic's coefficients from noisy data | `1.5 -> 1.5001`, `-0.8 -> -0.7988` |
| `xor/`         | a linear model against a two-layer one | linear `0.25`, two-layer `0` |
| `hessian/`     | the Hessian, its eigenvalues, a Newton step | one Newton step reaches the minimum |
| `checkpoint/`  | the memory/arithmetic trade-off | 50% less memory for 1.5x the arithmetic |
| `cse/`         | what the graph compiler saves | slots, folds, identities |

## Why `derivatives` matters

Every analytic gradient is printed next to a central difference of the same
function. A gradient engine that is subtly wrong can still produce
plausible-looking numbers, so the check that matters is against an
independent method rather than against a table of expectations.

The same argument applies inside the test suite, where every non-trivial
gradient is checked against *two* independent derivations: the closed form
and the forward-mode dual-number engine in `forward.mbt`, which shares no
code with the reverse pass.

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
