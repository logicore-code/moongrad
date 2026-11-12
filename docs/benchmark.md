# Benchmarks

Numbers below are the median wall-clock time per operation on
**MoonBit 0.1.20261104, native target, Linux x86_64, release build**.
Each is the median of 10 samples taken by `@bench.single_bench`, and
the harness measures in microseconds.

Reproduce with:

```bash
moon test --target native --release perf
```

## Core passes

All on a chain of `x -> x/2 + 0.1` repeated 400 times, which is 803 nodes
once the constants are hash-consed — the shape of a moderately deep
network's forward pass.

| Operation | Median | What it covers |
|---|---|---|
| build + compile the gradient | **559 µs** | constructing the graph and running the backward sweep once |
| forward pass | **14.7 µs** | constant propagation over 400 nodes |
| backward pass (re-evaluate the compiled gradient) | **27.0 µs** | the compiled gradient program, evaluated at a binding |
| backward pass with 8 checkpoint segments | **396 µs** | the same result, plus segment recomputation |
| one optimiser step, 6 samples, 8x4 network | **476 µs** | forward + backward + Adam, end to end |

## The tensor side

A 16-row batch through a 32×32 dense layer, measured separately
because the two sides have very different shapes.

| Operation | Median | What it covers |
|---|---|---|
| compile the tensor gradient | **122 µs** | the forward pass plus the backward sweep, once |
| re-evaluate the compiled gradient | **1 209 µs** | evaluating the gradient program at a binding |

Reading the two tables together is the point of this file.

- **The forward pass is cheap.** 14.7 µs for 803 nodes is about 18 ns per
  node, which is what a hash-consed flat array with a linear scan should
  cost.
- **On the scalar side, compiling is twenty times evaluating.** 559 µs
  against 27 µs. That is the price of the "the gradient is a program"
  design, and it is a price worth paying only once: the training loop
  compiles the gradient at construction and re-evaluates it every step.
- **On the tensor side the ratio is the other way round.** Compiling
  costs 122 µs and evaluating costs 1 209 µs, so a single evaluation is
  about ten times a compilation. The reason is the shape of the work:
  a tensor node is a whole array, so the backward sweep's job is to
  *emit* expressions — cheap — while the evaluation's job is to
  *compute* them, and computing a matmul is not cheap.
- **So "compile once" pays differently on the two sides.** On the scalar
  side it removes almost all of the per-step work. On the tensor side
  it removes one compilation out of every step's work, which is about a
  tenth of the step — and over a run of four hundred steps, that is
  four hundred compilations. The saving is real, but it is not the
  twenty-to-one the scalar number suggests, and quoting that number
  for the tensor side would be wrong.
- **Checkpointing costs about fifteen times the plain backward pass.**
  396 µs against 27 µs, because each of the 8 segments rebuilds its
  interior. That is the arithmetic side of the trade; the memory side
  is what `examples/checkpoint` prints, and the gradient is
  bit-identical either way.

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
3. **Reuse the buffers.** Every node of the gradient program
   allocates a tensor, and a graph of any size allocates a lot of
   them. A pool keyed by node slot would remove most of that
   pressure, and it is safe: the program is evaluated in node order,
   so a slot's value is always written before it is read. It is not
   done here because the pool has to be threaded through the
   evaluator's inner loop, which is the part of the library that has
   to stay simple enough to be obviously right.

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
- The optimiser-step row is the noisiest: it allocates the most, and on
  a shared machine it moved by a factor of two between runs of this
  suite. The ratio to the forward and backward rows is the stable part.
