# Benchmarks

Numbers below are the median wall-clock time per operation on
**MoonBit 0.1.20261104, native target, Linux x86_64, release build**.
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
| build + compile the gradient | **686 µs** | constructing the graph and running the backward sweep once |
| forward pass | **16.5 µs** | constant propagation over 400 nodes |
| backward pass (re-evaluate the compiled gradient) | **35.4 µs** | the compiled gradient program, evaluated at a binding |
| backward pass with 8 checkpoint segments | **494 µs** | the same result, plus segment recomputation |
| one optimiser step, 6 samples, 8x4 network | **642 µs** | forward + backward + Adam, end to end |

Reading the table:

- **The forward pass is cheap.** 16.5 µs for 400 nodes is about 40 ns per
  node, which is what a hash-consed flat array with a linear scan should
  cost.
- **The backward pass costs about twice the forward pass.** 35.4 µs
  against 16.5 µs. The backward pass is a second scan of a graph roughly
  twice the size of the primal — the gradient program has a node per
  local derivative rule — so the ratio is what the structure predicts.
- **Compiling is about 19x the cost of running the gradient once.**
  686 µs against 35.4 µs. That is the price of the "the gradient is a
  program" design, and it is a price worth paying only once: the
  training loop compiles the gradient at construction and re-evaluates it
  every step. A caller that differentiates afresh on every call would
  want the cheaper design; a training run never does.
- **Checkpointing costs about 14x the plain backward pass.** 494 µs
  against 35.4 µs, because each of the 8 segments rebuilds its interior.
  That is the arithmetic side of the trade; the memory side is what
  `examples/checkpoint` prints, and the gradient is bit-identical either
  way.

## Is it fast enough?

For the intended uses — embedded models, parameter fitting, research
prototypes, classroom demonstrations — yes. A model evaluated once per
request has a budget measured in milliseconds, and the whole evaluation
takes tens of microseconds.

It is not a numerical kernel. `moongrad` will not compete with a
hand-written kernel for a tight inner loop, and it is not trying to: the
value it offers is that the function being differentiated is
*user-supplied*, not MoonBit code the compiler could have optimised
directly.

## What would make it faster

In rough order of expected payoff:

1. **Simplify the gradient program.** The peephole pass runs on the
   primal but not on the gradient arithmetic emitted above it, and that
   arithmetic is where the node count doubles. Running the simplifier
   over the gradient program should remove a good fraction of it.
2. **Fuse the two scans.** The primal and the gradient live in one
   graph, so a single left-to-right pass could produce the forward
   values and the gradient values together, instead of evaluating the
   primal once to compile and again to run.
3. **Cache the primal's values between evaluations.** The training loop
   re-evaluates the primal at every step even though only the
   parameters changed. A cache keyed on node slot would cut the
   forward cost, at the price of the memory that the checkpointing
   module is about.

None of these are needed for the target use cases, and each would add
a layer between the user's expression and the machine. The
straightforward design is what keeps `moongrad` at 12k lines and
readable end to end.

## Caveats

- Numbers are sensitive to the MoonBit version and the host CPU. Treat
  them as relative order-of-magnitude, not absolute. The table above was
  taken on 0.1.20261104; an earlier table on 0.1.20260915 read about 20%
  higher across the board, which is the toolchain rather than the code.
- The benchmark harness itself runs inside the test binary, so the
  absolute figures include some harness overhead. The *ratios* between
  rows are the meaningful part.
- `grad_compile` builds a fresh graph each iteration on purpose: it is
  measuring the compile, not a cached reuse.
