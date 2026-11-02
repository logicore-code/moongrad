# moongrad

A reverse-mode automatic differentiation engine written in MoonBit.

`moongrad` compiles a program into a computation graph, then compiles the
*derivative* of that program into a second graph. The gradient is a first-class
value: you can print it, count its nodes, re-evaluate it at a different input,
or differentiate it again to get a Hessian.

```moonbit
import { "logicore-code/moongrad" }

fn main {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.sin(g.pow(x, 3.0))
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.5)
  let (grads, _program) = @moongrad.grad(g, b, f)
  println(grads[0])   // 3 * x^2 * cos(x^3) at x = 0.5
}
```

## Why the gradient is a program

Most autodiff engines accumulate numeric adjoints into a tape: the forward
pass stores every intermediate value, the backward pass reads them back and
pushes numbers into an array of partial derivatives. That works, and it has
two consequences worth noticing.

The first is that the gradient is *frozen at the point it was computed*. A
tape-based engine that asks for the gradient again after moving the weights
hands back the answer for the weights it used to have.

The second is that a tape has no second-order support without a second
engine. Differentiating a gradient means differentiating numbers that were
already collapsed, which loses the structure you would need.

`moongrad` takes the other route. The backward sweep emits a *program* whose
nodes are `d(f) / d(primal_slot)`, built with the same builder and the same
simplifier as the primal. Two things follow:

- **The gradient is re-evaluable.** It is a function of its inputs, so it can
  be evaluated at any point. The training loop in `train.mbt` compiles once
  and re-evaluates every step.
- **Second-order derivatives are nearly free.** The Hessian entry
  `H[i][j]` is `d²f/dx_j dx_i`, which is another call to the same compiler
  with the first sweep's output as the target. `hessian.mbt` is eleven lines
  long because of it.

The test suite checks every non-trivial gradient against two independent
implementations — the closed form, and a forward-mode dual-number engine that
shares no code with the reverse pass.

## What is in the box

| Module | Responsibility |
|---|---|
| `node.mbt` | The IR. A flat array of slots; edges always point to lower indices. |
| `graph.mbt` | The builder: hash-consing, peephole simplification, statistics. |
| `simplify.mbt` | The peephole rules and the local derivative tables. |
| `forward.mbt` | Constant propagation, plus a forward-mode dual-number engine. |
| `reverse.mbt` | The gradient compiler. |
| `hessian.mbt` | Second-order derivatives, Cholesky, a damped Newton step. |
| `checkpoint.mbt` | Segment-recomputation gradient checkpointing. |
| `tensor.mbt` | A small row-major matrix type. |
| `nn.mbt` | Dense layers, activations, losses. |
| `optim.mbt` | SGD, momentum, Adam. |
| `train.mbt` | The training loop. |

151 unit tests, 0 compiler warnings.

## The language subset

Special forms: `quote`, `if`, `define` (both forms), `set!`, `lambda`, `let`,
`let*`, `letrec`, named `let`, `cond`, `begin`.

Parameters are first-class: a `Param` is a vector of scalars that the
gradient compiler differentiates alongside the variables, so the optimiser
steps on both from one gradient program.

Primitives: `+ - * /`, `= < > <= >=`, `null? pair? number? symbol? string?
boolean? eq?`, `cons car cdr list length append reverse`, `display print
number->string`, and the unary operators `neg exp log sqrt sin cos tanh sigmoid
softplus relu`.

## Examples

Each directory under `examples/` is an executable package:

```bash
moon run examples/derivatives  --target native   # derivative table vs finite differences
moon run examples/polyfit      --target native   # recover a polynomial's coefficients
moon run examples/xor          --target native   # a linear model vs a two-layer one
moon run examples/hessian      --target native   # Hessian, eigenvalues, a Newton step
moon run examples/checkpoint   --target native   # the memory/arithmetic trade
moon run examples/cse          --target native   # what the graph compiler saves
```

Two of them are worth a second look.

`examples/derivatives` prints each analytic gradient next to a central
difference of the same function. A gradient engine that is subtly wrong can
still produce plausible-looking numbers, so the check that matters is against
an independent method.

`examples/checkpoint` prints what each segment count costs in memory and in
arithmetic, and asserts that the gradient is bit-identical in every
configuration. A memory strategy that quietly computes something different is
worse than no memory strategy.

## Building

```bash
moon test          # 151 unit tests
moon check         # 0 warnings, 0 errors
moon fmt           # format
moon build         # build for the preferred target (wasm)
```

The library targets `wasm`, `wasm-gc`, `js` and `native` from the same
source with no conditional compilation.

## Design notes

`docs/architecture.md` walks through the three decisions that shape
everything else: why the IR is a flat array, why closures are encoded as
s-expressions rather than a MoonBit closure, and why the backward sweep
emits expressions rather than numbers.

`docs/math.md` derives the derivative rules the compiler emits and explains
where each one comes from.

## License

Apache-2.0. See `LICENSE`.
