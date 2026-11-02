# Benchmarks

Numbers below are the median wall-clock time per operation on
**MoonBit 0.1.20260915, native target, Linux x86_64, release build**.
Each is the median of 10 samples taken by `@bench.single_bench`.

Reproduce with:

```bash
moon test --target native --release perf
```

## Core passes

All on a 400-node computation — a chain of `x -> x/2 + 0.1` repeated 200
times, which is the shape of a moderately deep network's forward pass.

| Operation | Median | What it covers |
|---|---|---|
| build + compile the gradient | **862 µs** | constructing the graph and running the backward sweep once |
| forward pass | **21 µs** | constant propagation over 400 nodes |
| backward pass (re-evaluate the compiled gradient) | **45 µs** | the compiled gradient program, evaluated at a binding |
| backward pass with 8 checkpoint segments | **699 µs** | the same result, plus segment recomputation |
| one optimiser step, 6 samples, 8x4 network | **847 µs** | forward + backward + Adam, end to end |

Reading the table:

- **The forward pass is cheap.** 21 µs for 400 nodes is about 50 ns per node,
  which is what a hash-consed flat array with a linear scan should cost.
- **The backward pass costs about twice the forward pass.** 45 µs against
  21 µs. The backward pass is a second scan of a graph roughly twice the
  size of the primal — the gradient program has a node per local derivative
  rule — so the ratio is what the structure predicts.
- **Compiling is 40x the cost of running the gradient once.** 862 µs against
  45 µs. That is the price of the "the gradient is a program" design, and
  it is a price worth paying only once: the training loop compiles the
  gradient at construction and re-evaluates it every step. An interactive
  session that differentiates on every keystroke would want the cheaper
  design; a training run never does.
- **Checkpointing costs 15x the plain backward pass.** 699 µs against
  45 µs, because each of the 8 segments rebuilds its interior. That is
  the arithmetic side of the trade; the memory side is what
  `examples/checkpoint` prints, and the gradient is bit-identical either
  way.

## Is it fast enough?

For the intended uses — embedded DSLs, rule engines, parameter fitting,
classroom demonstrations — yes, and by a wide margin. A rule predicate
evaluated once per HTTP request has a budget measured in milliseconds, and
the whole evaluation takes tens of microseconds.

It is not a numerical kernel. `moongrad` will not compete with a
hand-written gradient for a tight inner loop, and it is not trying to:
the value it offers is that the function being differentiated is
*user-supplied Scheme*, not MoonBit code the compiler could have
optimised directly.

## What would make it faster

In rough order of expected payoff:

1. **Inline caching for primitives.** Every `+` on two scalars currently
   costs an `SExp::match` round trip. A primitive-node fast path would
   take a large bite out of both passes.
2. **A bytecode VM.** Walking the AST directly means re-decoding the
   operator on every visit. Compiling to a small register machine and
   interpreting that would cut both passes substantially, at the cost of
   a new back end.
3. **Tail-call optimisation.** Every call allocates a frame. The
   self-recursive shape that dominates these workloads would get
   materially faster.

None of these are needed for the target use cases, and each would add a
layer between the user's expression and the machine. The AST-walking
design is what keeps `moongrad` at 1.6k lines and readable end to end.

## Caveats

- Numbers are sensitive to the MoonBit version and the host CPU. Treat
  them as relative order-of-magnitude, not absolute.
- The benchmark harness itself runs inside the test binary, so the
  absolute figures include some harness overhead. The *ratios* between
  rows are the meaningful part.
- `grad_compile` builds a fresh graph each iteration on purpose: it is
  measuring the compile, not a cached reuse.
