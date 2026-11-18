# Architecture

Three decisions shape everything else in `moongrad`. This document is about
those three, and about what they cost.

## 1. The IR is a flat array of slots

A node is referred to everywhere else by its index into `Graph::nodes`:

```moonbit
pub(all) enum Node {
  Const(Double)
  Var(Int32)
  ParamGet(Int32, Int32)
  Unary(UnaryOp, Int32)
  Binary(BinaryOp, Int32, Int32)
  PowBy(Double, Int32)
  ReluDeriv(Int32)
}
```

The important property is not the representation but the invariant it
supports: **a node can only be interned after its children exist**, so every
edge points from a higher index to a lower one.

That single invariant removes two whole categories of machinery:

- **The forward pass is a left-to-right scan.** No topological sort, no
  worklist, no visiting-state array. When the sweep reaches a slot, its
  operands are already in the value array.
- **The backward pass is a right-to-left scan.** No scheduler, no priority
  queue, no "which node still has pending adjoints" bookkeeping. When the
  sweep reaches a slot, every consumer of it has already been processed.

It also makes the graph a DAG for free, because the builder hash-conses:
interning an identical node twice returns the first slot, so two
structurally identical subexpressions are one node with two parents.

The cost is that the IR cannot express anything with more than two operands
without an extra convention, and that a node's children have to be
contiguous in memory for the sweeps to be cache-friendly. Neither has
mattered so far; a compiler backend that wanted a register machine would
have to change the representation rather than the passes.

## 2. Two engines, one set of rules

There are two gradient compilers in this repository: a scalar one and a
tensor one. They are separate implementations, and it is worth being
explicit about why, because the obvious answer — "the scalar one is a
prototype, the tensor one grew out of it" — is not what happened.

The scalar engine is the **rank-0 case**. Its node holds a `Double`; the
tensor engine's holds a whole array. Both are the same idea applied at
different ranks, and both emit their gradients the same way, so a rule
fixed in one is fixed in the other.

What they genuinely share is the derivative *rule*, not the code. The
scalar backward sweep emits expressions in a scalar graph and the
tensor one emits expressions in a tensor graph, so each carries its own
rules. `simplify.mbt` holds a third copy — the numerical table
`unary_derivative`, `binary_dleft` and `binary_dright` — which is what
the forward-mode engine differentiates against, because it needs a
number rather than an expression.

Keeping the two from drifting is therefore a matter of testing rather
than of sharing. `engine_agreement_test.mbt` compares what the two
engines compute for every operator they have in common, and the
finite-difference tests check both against a derivation neither of them
uses.

The tensor engine has one thing the scalar one does not need, and it is
the reason its gradient program is not a simple multiple of the primal's
size: **broadcasting**. When an operand was repeated to reach the
operand's shape, its gradient comes back at the broadcast shape and has to
be summed down to the operand's own shape. The `Adjust` node is that
operation, and it is the tensor engine's own addition to the vocabulary.

## 3. The backward sweep emits expressions, not numbers

The sweep itself is short:

```
adj[target] := Const(1)
for i := n-1 downto 0:
    if adj[i] is not the zero constant:
        for each child c of nodes[i] with local derivative d:
            adj[c] += adj[i] * d
```

where `d` is emitted into the graph rather than computed as a number.

A tape-based engine has no choice: it evaluates `d/dx sin(x)` to
`cos(1.7)` and stores the number. `moongrad` emits a `cos` node applied to
the primal's own `x` node instead.

This costs a few more nodes and buys two things.

**The gradient is re-evaluable.** A tape returns the answer for the point it
was computed at. `moongrad` returns a program, so the trainer can evaluate
it at whatever weights the optimiser has reached. The test suite caught this
directly: a gradient compiled at `x = 1` and evaluated at `x = 5` returned 3
instead of 75.

**Second-order derivatives need no second engine.** The Hessian entry
`H[i][j]` is `d²f/dx_j dx_i`, which is another call to the same compiler
with the first sweep's output as the target. the `hessian` function in
`hessian.mbt` is about a dozen lines long because of it.

### The tensor engine does the same thing

The tensor side was originally written the textbook way: the backward sweep
accumulated numeric adjoints into an array of tensors. It worked, and it
was quietly inconsistent with the scalar side — the scalar gradient was a
program and the tensor gradient was not, so the two engines disagreed
about what "compile once" meant.

The tensor sweep now emits expressions too. Every adjoint is a node in the
graph, and the local derivatives are the same expressions the scalar engine
emits, widened to tensors. Two node types carry the weight:

- `Adjust` moves a value between shapes: up by broadcasting when the target
  has more axes, down by summing when it has fewer. Broadcasting in reverse
  mode is exactly this operation, and it is its own inverse in the sense
  that matters — the gradient of a broadcast is a sum, and the gradient of a
  sum is a broadcast.
- `ReluDeriv` is the same step function the scalar side uses, on whole
  tensors.

The gradient program for a tensor graph is therefore the same kind of
object as the scalar one: a graph that can be evaluated at any binding.
`examples/reuse` compiles a tensor gradient once and evaluates it at four
inputs, two of them on the far side of a ReLU's kink.

### A bug this design caught

The scalar engine's parameter gradients were read out of a *frozen* copy of
the primal's values, so a gradient program evaluated at a different binding
than the one it was compiled against produced the gradient of the wrong
function. The variable gradients were correct; the parameter gradients were
one step behind. Training masked it, because the training loop refreshed
the primal before every step, but the API was a trap.

Emitting expressions is what made the bug visible: once every local
derivative is recomputed at evaluation time, the primal has to be
recomputed at evaluation time too, and the frozen array had nowhere to hide.

### The gradient program lives in the same graph

The gradient's nodes go into the same array as the primal's, above it.
That is not tidiness; it is what makes evaluation a single pass. The
primal's nodes occupy the low slots and the gradient's the high ones, so
one left-to-right scan evaluates the primal and then the gradient, with
the gradient's nodes reading the primal's values out of the same array.

It also means the gradient program can be differentiated again. The
second sweep sees an ordinary graph — the primal's nodes followed by the
gradient's — and the rules that apply to it are the same rules. Nothing
in the compiler knows it is looking at a gradient.

The sweep works on a **copy** of the primal rather than the primal
itself. The copy is made by re-interning every node, which for a
hash-consed graph is nearly free: each node finds its own slot, so the
copy has exactly the same shape. Mirroring rather than sharing is
deliberate — sharing one node array would make it impossible to keep the
primal after differentiating it, and the primal is worth keeping, because
it is what the caller differentiates next.

## Memory

A tape-based reverse sweep needs the forward value of every node, because
each local derivative rule reads its operand's value. A textbook tape
stores them all: one backward pass through an `n`-node program costs `O(n)`
memory.

`moongrad` does not keep a tape. Because the sweep emits expressions
rather than reading operand values, it holds nothing but the adjoint array.
The tape is gone, and with it the reason the sweep had to hold anything.

The memory did not disappear; it moved to the evaluation. Evaluating the
gradient program holds a value per node, and that is the quantity
`checkpoint.mbt` is about.

`checkpoint.mbt` evaluates the gradient program in segments — the scan is
partitioned into `k` contiguous runs — and measures what the evaluation
actually costs. The answer is that the segmentation does not pay here, and
the reason is worth recording.

A tape's interior depends on its segment's boundary and on nothing else, so
dropping it and rebuilding it later costs one re-evaluation. A gradient
program's nodes do not have that property: every local derivative is an
expression over the primal's own nodes, so a gradient node refers back into
the primal, at an arbitrary index. Keeping only the boundaries would mean
dropping primal values the program still needs, and re-deriving a dropped
value is not one evaluation either: the primal's nodes are in topological
order, so re-deriving one means re-deriving its predecessors, and the cost
of a rebuild is the depth of the primal rather than the width of a segment.

So the `O(k + w)` bound belongs to the technique and not to this program.
The module measures what the evaluation actually costs, and the measurement
is the finding: holding a gradient program costs *more* than a tape would,
not less, because the program is bigger than the primal it came from — and
the cost does not depend on how the program is cut up.

The report says the same thing in numbers. `peak_live` is the working set
the evaluation really holds: the primal's values and the gradient program's
own, all of them live at once, because a gradient node may read a primal
value at any index and no value can be dropped while it might still be
read. `naive_live` is what a tape would hold, one value per primal node.
`gradient_evals` is the gradient program's own node count — the arithmetic
it spends on top of the forward pass the caller already paid for.
`reduction` is the working set against the tape's, and it is negative — and
it falls below `-1` whenever the gradient program is bigger than the
primal, because the ratio is not confined to `[-1, 1]`: it says which side
of the tape's working set the evaluation falls on, not how much of a saving
there is. `arithmetic_cost` is the evaluation count as a multiple of the
forward pass, and it is the same for every segmentation.

What the module establishes is the part that has to be true before the
arithmetic is worth discussing: the segmented evaluation returns the same
numbers as the plain one, in every configuration.

The conclusion is that the gradient-program design answers the tape's
memory question a different way. There is no tape to hold between the
forward pass and the backward pass, so there is nothing to checkpoint. The
evaluation holds one value per node — the same order as the tape held —
but only while it runs.

`choose_segment_count` takes a target *width* rather than a segment count,
so "keep the window under 256 nodes" behaves the same way on a graph of any
size. It is the count the textbook sweep would use for that window width;
here it partitions the scan, and the partition does not lower the working
set.

## The simplifier

The peephole pass runs before interning, so two computations that differ
only in algebraic form still land on the same slot.

Three families of rule:

| Family | Examples | Effect |
|---|---|---|
| annihilator | `x*0 → 0`, `x-x → 0` | drops the whole subtree, but only when the other operand is a literal known to be finite |
| identity | `x+0 → x`, `x*1 → x`, `x/1 → x` | drops the operator |
| fold | both operands literal | evaluates now |

The conservatism is the point. A simplifier that is too eager changes the
floating-point semantics of the caller's program, and a training run that
converges to a slightly different minimum because `x*0` was rewritten
before the rounding had a chance to accumulate is a genuinely unpleasant
thing to debug. Every rule here is exact in IEEE-754 for the operands it
accepts, and nothing is rewritten when an operand is non-finite. Anything
that would evaluate outside the finite reals is left for the forward pass to
report with the operands attached.

`Stats` counts what happened, so the optimisation is measurable rather than
asserted: `cse_hits`, `identities`, `folds`, `annihilated`, `requests`.

## Design decisions

These are choices, with reasons, rather than gaps.

- **The graph is compiled for a set of shapes.** The backward rules bake
  the operand shapes into the program, because the shapes are what say
  how a broadcast has to be summed back down. A compiled gradient
  records the input shapes it was compiled for and checks them at
  evaluation, so a mismatch raises rather than quietly computing the
  gradient of a different function.
- **Intermediate values are not written in place.** In-place updates
  would save memory and would break re-evaluation: the first evaluation
  would dirty the buffers and the second would read corrupted values.
  Re-evaluability is the property the whole design is for, so the
  buffers stay and `checkpoint.mbt` answers the memory question
  instead.
- **The mini-batch size is fixed at compile time.** The batch is an
  input, so the graph does not change between batches; but the batch
  axis has a length, and that length is part of the graph's shapes.
  A different batch size is a different graph, and therefore a
  different compilation. A training run does not change its batch
  size, so this costs nothing in practice.
- **Two engines, and a test that they agree.** The scalar engine is the
  rank-0 case of the tensor one, not a prototype it grew out of. They
  are separate code with separate backward sweeps, so what keeps a
  rule fixed in one from being un-fixed in the other is
  `engine_agreement_test.mbt`, which compares what the two compute
  for every operator they have in common. What they do share is the
  table that says which literal is the identity of which operator,
  because the tensor engine's simplifier asks it the same question
  the scalar one does.
- **Two routes to a second-order step.** The dense Hessian is exact
  and simple when the parameter count is small. The matrix-free route
  — a Hessian-vector product plus conjugate gradient — is what
  scales, and the two agree to within one unit in the last place.
