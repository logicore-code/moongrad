# The derivative rules

Where each rule the compiler emits comes from, and what it costs.

## The chain rule, in the form the compiler uses

For a scalar function `f` and a program that computes it, the derivative
with respect to an input `x` is a sum over all the paths from `x` to `f` of
the product of the local derivatives along each path:

```
df/dx = Σ_paths  Π_edges (∂(edge output) / ∂(edge input))
```

The backward sweep computes exactly this, one node at a time, by
accumulating adjoints: `adj[i]` is the derivative of the final output with
respect to the value of node `i`.

The sweep is right-to-left because the graph's edges point to lower indices,
so when the sweep reaches a node, every consumer of it has already been
processed and its adjoint is final.

## Unary operators

| Operator | Rule | Note |
|---|---|---|
| `neg(x)` | `-1` | constant |
| `exp(x)` | `exp(x)` | the function is its own derivative |
| `log(x)` | `1/x` | domain `x > 0` |
| `sqrt(x)` | `1/(2*sqrt(x))` | domain `x >= 0` |
| `sin(x)` | `cos(x)` | |
| `cos(x)` | `-sin(x)` | |
| `tanh(x)` | `1 - tanh(x)^2` | |
| `sigmoid(x)` | `s*(1-s)`, `s = sigmoid(x)` | |
| `softplus(x)` | `sigmoid(x)` | `softplus(x) = log(1 + e^x)` |
| `relu(x)` | `1` if `x > 0` else `0` | see below |

Every one of these is emitted as an *expression* over the primal's nodes,
not as a number. `d/dx sin(x)` becomes a `cos` node applied to the primal's
`x` node, so the gradient recomputes correctly at a different binding.

The one exception is ReLU. Its derivative is discontinuous — zero below the
kink, one above — and the IR has no comparison with which to say "if the
runtime value is positive". The compiler therefore reads the primal value
and emits a literal 0 or 1.

That is a real limitation rather than a shortcut, and it is the price of
keeping the IR free of control flow: **a ReLU's gradient is correct at the
point it was compiled and has to be recompiled for another point.** A
system that wanted a re-evaluable ReLU gradient would need a comparison in
the IR, which means a conditional node and a forward pass with branches.

The convention at exactly zero is 0, which is the usual one. It keeps a unit
that lands exactly on the boundary from producing a gradient of 0.5 and
stalling.

## Binary operators

For `z = a op b`:

| Operator | `∂z/∂a` | `∂z/∂b` |
|---|---|---|
| `a + b` | `1` | `1` |
| `a - b` | `1` | `-1` |
| `a * b` | `b` | `a` |
| `a / b` | `1/b` | `-a/b²` |

`a - b` is not symmetric, so the simplifier only fires the identity rule
when the *right* operand is the identity. `0 - b` is `-b`, which is a
rewrite the caller has to ask for.

## Powers

`d/dx x^c = c * x^(c-1)`, for a constant exponent `c`.

The exponent rides along inside the node rather than in an operand slot,
so the rule needs no table lookup and `PowBy` stays at arity one. `x^2`
gets its own node shape because it is the dominant term in a quadratic
model and having it lets the simplifier recognise `x^2 - x^2` as zero
without evaluating either square.

## Losses

**Mean squared error.** For `L = (1/n) * Σ (pred_i - y_i)^2`:

```
dL/dpred_i = 2 * (pred_i - y_i) / n
```

Divided by the batch count and not by `2 * batch`, so the reported number
is the actual mean squared error and can be compared against a variance.

**Binary cross-entropy.** Written as

```
L = softplus(z) - z*y
```

rather than the textbook `-y*log(sigmoid(z)) - (1-y)*log(sigmoid(-z))`. The
textbook form computes a logarithm of something that underflows to zero once
`|z|` exceeds about 40, and then returns `-inf` for a perfectly confident
*correct* prediction. The softplus form has no such cliff: `softplus` is
evaluated as `max(z,0) + log(1 + e^-|z|)`, which stays finite everywhere.

Its derivative is

```
dL/dz = sigmoid(z) - y
```

which is also the numerically stable form: it never multiplies a
probability by a logarithm.

**The sigmoid/logit pairing matters.** The loss wants the *logit*, not the
probability. Feeding a post-sigmoid value into `softplus(z) - z*y` squashes
the gradient exactly where it needs to be largest. `examples/xor` uses a
linear output layer for exactly this reason, and applies the sigmoid only
when printing the prediction.

## Second-order derivatives

The Hessian entry `H[i][j]` is `d²f/dx_j dx_i`. With the gradient as a
graph, this is another call to the gradient compiler with the first
sweep's output node as the target:

```
for i in 0..n:
    (row, _) = grad(gradient_graph, binding, gradient_graph.var_adj[i])
    for j in 0..n:
        H[i][j] = row[j]
```

The off-diagonal entries are where the product rule shows up. For
`f = exp(x*y)`:

```
df/dx   = y e^(xy)
d²f/dxdy = (1 + xy) e^(xy)
```

The plain `1` rides along from the product rule. An engine that only
propagates through the exponential gets the exponential part right and
still fails.

## Numerical differentiation, as a check

Central differences give an independent estimate:

```
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
```

With `h = 1e-5` and `f` evaluated in double precision this gives about
seven correct digits for the smooth functions in
`examples/derivatives`, which is plenty to catch a wrong derivative rule.
It is also far enough from the catastrophic cancellation that a forward
difference `(f(x+h) - f(x))/h` would suffer at.

The tests use it as a third opinion alongside the closed form and the
forward-mode engine. Three derivations that agree is strong evidence;
two is suggestive.
