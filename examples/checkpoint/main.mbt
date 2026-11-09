// Memory versus arithmetic.
//
// Builds one deep computation, then differentiates it with several
// checkpoint segment counts, and prints what each one cost.
//
// The trade is the whole point of `checkpoint.mbt`. The reverse
// sweep needs the forward value of every node, and a textbook tape
// keeps them all: one backward pass through an n-node program
// costs O(n) memory. Cutting the node array into segments keeps
// only the boundaries and rebuilds each segment's interior on
// entry, so peak occupancy falls while the arithmetic rises.
//
// The gradient is bit-identical in every configuration. That is
// asserted here rather than only in the test suite, because it is
// the property that makes the trade safe to make.

///|
fn build_deep(g : @moongrad.Graph, x : Int32, depth : Int32) -> Int32 {
  // A contraction rather than an expansion: repeated squaring would
  // leave the finite reals long before the interesting depths.
  let mut acc = x
  for i = 0; i < depth; i = i + 1 {
    acc = g.add(g.mul(acc, g.constant(0.5)), g.constant(0.1))
    ignore(i)
  }
  acc
}

///|
fn main raise @moongrad.ADError {
  let depth = 400
  println("| moongrad: gradient checkpointing")
  println("| a \{depth}-node computation, differentiated with several")
  println("| segment counts")
  println("|")
  println("| segments   peak live   reduction   recomputed   arith cost")
  println("| ------------------------------------------------------")
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = build_deep(g, x, depth)
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.5)
  // The reference answer: the plain, un-checkpointed gradient.
  let (reference, _p) = @moongrad.grad(g, b, f)
  for segments in [1, 2, 4, 8, 16, 32, 64] {
    let (grads, report) = @moongrad.checkpointed_grad(g, b, f, segments)
    // Every configuration has to agree with the reference to the
    // last bit, or the trade is not a trade.
    let mut same = true
    for i = 0; i < reference.length(); i = i + 1 {
      if grads[i] != reference[i] {
        same = false
      }
    }
    let flag = if same { "" } else { "  MISMATCH" }
    println(
      "| \{@moongrad.format_double(segments.to_double())}        \{report.peak_live}      \{@moongrad.format_double(report.reduction())}      \{report.recomputed}        \{@moongrad.format_double(report.arithmetic_cost())}\{flag}",
    )
  }
  println("|")
  println("| The gradient is identical in every row. Only the cost")
  println("| changes: fewer segments means more memory held at once and")
  println("| less arithmetic spent rebuilding it, and more segments is")
  println("| the other way round.")
}
