# Architecture

Three decisions shape everything else in `moongrad`. This document is about
those three, and about what they cost.

## 1. The IR is a flat array of slots

A node is referred to everywhere else by its index into `Graph::nodes`:

```moonbit
pub(all) enum Node {
  Const(Double)
  Var(Int)
  ParamGet(Int, Int)
  Unary(UnaryOp, Int)
  Binary(BinaryOp, Int, Int)
  PowBy(Double, Int)
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

## 2. Closures are s-expressions, not MoonBit closures

A closure is encoded as

```
(params body . env_marker)
```

where `env_marker` is a symbol like `__env__:42` that resolves to the
captured `Env` through a global table.

The obvious alternative — a MoonBit closure capturing an `Env` — does not
work, and the reason is worth stating because it is easy to get wrong. A
MoonBit closure that captures an environment would capture it *by value* at
the moment the closure is constructed. But the environment has to change: a
`define` inside the body binds a new name, and the closure needs to see it.

The marker indirection solves this. The marker names a slot in a global
table, the table entry points at the current environment, and the closure
holds only the marker's name. When the environment changes, the table entry
is repointed, and every closure that captured that marker sees the change.

The table is also what makes recursive `define` work. See below.

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
with the first sweep's output as the target. `hessian.mbt` is eleven lines
long because of it.

### Recursive define, and the marker rebinding

`(define (fact n) (if (<= n 1) 1 (* n (fact (- n 1)))))` needs the body to
refer to `fact`.

The build is:

1. Allocate a fresh marker `m`.
2. Build the closure `(params body . m)`.
3. Bind `fact` to that closure, producing `env'`.
4. **Re-point** `m` at `env'`.

After step 4 the closure's marker resolves to an environment that contains
the closure itself under the name `fact`, so the recursive call resolves and
the cycle terminates.

`letrec` uses the same idea with a sweep: every marker allocated during the
evaluation of the bindings is re-pointed at the patched environment
afterwards. A per-marker approach was tried first and does not work — each
`make_closure` inside a value's body allocates its own marker, so rebinding
the placeholder leaves the value's captured environment out of sync.

## Memory

The reverse sweep needs the forward value of every node, because each local
derivative rule reads its operand's value. A textbook tape stores them all:
one backward pass through an `n`-node program costs `O(n)` memory.

`checkpoint.mbt` exploits the same index ordering that made the sweep
simple. Children sit at lower indices, so a right-to-left pass can reach any
window of the forward computation on demand — it needs each value shortly
before consuming it and never again.

Cut the node array into `k` segments, keep only the boundary values, and
rebuild each segment's interior on entry. Peak occupancy drops from `O(n)`
to `O(k + w)` for segments of width `w`, at the cost of re-evaluating every
interior node once. That is the trade Chen et al. described for gradient
checkpointing in 2016, applied to a compiler rather than to an interpreter.

`choose_segment_count` takes a target *width* rather than a segment count,
so "keep the window under 256 nodes" behaves the same way on a graph of any
size. Many narrow segments buy memory with arithmetic and approach a 2x
cost; few wide ones do the opposite. The gradient is bit-identical in every
configuration, which is what makes the trade safe to make.

## The simplifier

The peephole pass runs before interning, so two computations that differ
only in algebraic form still land on the same slot.

Three families of rule:

| Family | Examples | Effect |
|---|---|---|
| annihilator | `x*0 → 0`, `x-x → 0` | drops the whole subtree |
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

## What is deliberately absent

- **No bytecode VM.** The AST walk is direct. A register machine would be
  faster and would change the IR rather than the passes.
- **No tape.** The gradient is a program, so there is nothing to free.
- **No mini-batch support in the trainer.** The graph covers the whole
  dataset as one full batch, because a graph per batch size would have to be
  rebuilt and a graph per step would throw away the compile-once property.
- **No `delay`/`force`.** Lazy evaluation would need the IR to grow a
  suspension type, and the lazy-stream example demonstrates how to get the
  same effect from closures alone.
