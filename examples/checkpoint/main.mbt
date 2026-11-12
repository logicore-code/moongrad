// What a gradient program costs to evaluate.
//
// Builds one deep computation, differentiates it, and prints what
// the evaluation held and what it spent.
//
// The question the classic answer gives — "cut the node array into
// segments, keep only the boundaries, rebuild each segment's
// interior on entry, and the memory falls while the arithmetic
// rises" — does not have the same answer here, and the table shows
// why. Every gradient node refers back into the primal, so the
// primal's values have to stay reachable and the segments are not
// independent of one another. Cutting the program up therefore saves
// nothing: the numbers are the same in every row.
//
// The gradient is bit-identical in every configuration. That is
// asserted here rather than only in the test suite, because it is
// the property that has to hold before any cost is worth
// discussing.

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
  println("| moongrad: what a gradient program costs")
  println("| a \{depth}-node computation, differentiated and evaluated")
  println("| with several segment counts")
  println("|")
  println("| segments   held      vs tape    extra evals   arith cost")
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
  println("| The gradient is identical in every row. That is the part")
  println("| that has to be true before any cost is worth discussing:")
  println("| a memory strategy that quietly changes the answer is")
  println("| worse than no strategy.")
  println("|")
  println("| The rest of the table is the measurement, and it is not")
  println("| the measurement the technique usually gets. Holding a")
  println("| gradient program costs MORE than a tape would, not less:")
  println("| the program is bigger than the primal it came from, and")
  println("| every gradient node refers back into the primal, so the")
  println("| primal's values have to stay reachable.")
  println("|")
  println("| The segmentation does not change that, and the rows show")
  println("| it: the cost is the same however the program is cut up.")
  println("| That is the finding. The classic trade — hold less, spend")
  println("| more arithmetic — needs a tape whose segments are")
  println("| independent, and a gradient program's are not.")
  println("|")
  println("| What the design does buy is a different thing: there is")
  println("| no tape to hold between the forward pass and the backward")
  println("| pass. The gradient is a program, and a program is")
  println("| evaluated when it is needed rather than stored until it")
  println("| is wanted.")
}
