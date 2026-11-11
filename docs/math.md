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
| `relu(x)` | `ReluDeriv(x)` | see below |
| `silu(x)` | `s + x s (1-s)`, `s = sigmoid(x)` | the sigmoid linear unit, also called swish |
| `gelu(x)` | see below | the tanh approximation |

Every one of these is emitted as an *expression* over the primal's nodes,
not as a number. `d/dx sin(x)` becomes a `cos` node applied to the primal's
`x` node, so the gradient recomputes correctly at a different binding.

ReLU is the one rule that is not a smooth function of the operand, and it
gets its own node for the same reason: `ReluDeriv` is evaluated where the
gradient is evaluated, not where the gradient was compiled.

```
ReluDeriv(x) = 1 if x > 0 else 0
```

The comparison is inside the node. That is the difference between a branch
decided once, at compile time, and a branch decided every time the gradient
runs — and it is the difference between a gradient that is right at one
point and a gradient that is right everywhere.

The convention at exactly zero is 0, which is the usual one. It keeps a unit
that lands exactly on the boundary from producing a gradient of 0.5 and
stalling.

The second derivative of ReLU is therefore zero almost everywhere, because
the derivative of a step function is zero away from the kink. At the kink
itself the second derivative does not exist, and no convention makes it
well defined.

GELU's derivative is longer but has nothing discontinuous in it. With
`u = c (x + 0.044715 x^3)` and `c = sqrt(2/pi)`, the value is
`0.5 x (1 + tanh u)` and the derivative is

```
0.5 (1 + tanh u) + 0.5 x (1 - tanh^2 u) u'
```

with `u' = c (1 + 3 * 0.044715 x^2)`. The compiler emits that as an
expression over the primal's own nodes, so it is recomputed rather
than remembered, like every other rule here.

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

### The tensor side

The tensor engine produces the same object over a larger index space.
An input slot holds a tensor rather than a scalar, so the Hessian is over
the *flattened* input space: the elements of slot 0 first, then those of
slot 1, and so on.

The construction is the scalar one, run one element at a time:

```
for each input slot s, for each element e of s:
    element = Select(gradient_program.input_adj[s], e)   # d f / d x_{s,e}
    second = compile_gradient(gradient_program, element)
    for each input slot s2, for each element e2 of s2:
        H[flat(s2) + e2][flat(s) + e] = second.input_grads[s2][e2]
```

`Select` is a node, so `d f / d x_{s,e}` is a node, and the second sweep
is an ordinary backward pass over an ordinary tensor graph. There is no
second-order special case anywhere in the compiler.

`param_hessian` and `mixed_hessian` are the same construction with the
parameter space in place of the input space. The mixed block,
`d²f/dw_i dx_j`, is the one a Gauss-Newton step needs.

### Hessian-vector products

The dense Hessian is O(n^2) in memory and costs one extra sweep per
parameter element. A Hessian-vector product is O(n) and costs one
sweep:

```
H v = grad( g . v )
```

where `g` is the gradient. The Hessian is the transpose of the
Jacobian of `g`, so

```
(H v)_i = d/dx_i ( sum_j d f/dx_j v_j )
```

which is the gradient of the scalar `g . v`. So a Hessian-vector
product is the ordinary backward sweep pointed at a different node —
the directional derivative built out of the gradient program's own
nodes, and then differentiated.

The direction is an *input* to the compiled program rather than a
constant inside it, so one compilation serves every direction. That is
what makes a conjugate-gradient iteration affordable: the operator is
applied to a new direction at every step, and each application is one
evaluation rather than one compilation.

`newton_cg_step` uses this to solve

```
(H + damping I) dx = -g
```

without forming `H`. The damping is what makes the system positive
definite; the exact Hessian of a non-convex objective often is not.

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
