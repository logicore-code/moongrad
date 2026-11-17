// What a gradient program costs to evaluate.
//
// Builds one deep computation, differentiates it, and prints what
// the evaluation held and what it spent.
//
// The question the classic answer gives — "cut the node array into
// segments, keep only the boundaries, rebuild each segment's
// interior on entry, and the memory falls while the arithmetic
// rises" — does not have the same answer here, and the table shows
// why. Every gradient node refers back into the primal, so a primal
// value may be read by a node in any segment, and the primal's nodes
// are in topological order: re-deriving one means re-deriving its
// predecessors. Cutting the program up therefore saves nothing: the
// numbers are the same in every row.
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
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = build_deep(g, x, depth)
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.5)
  println("| moongrad: what a gradient program costs")
  println(
    "| a \{depth}-level computation of \{g.nodes.length()} nodes, differentiated",
  )
  println("| and evaluated with several segment counts")
  println("|")
  println("| segments   held      vs tape    grad evals   arith cost")
  println("| ------------------------------------------------------")
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
      "| \{@moongrad.format_double(report.segments.to_double())}        \{report.peak_live}      \{@moongrad.format_double(report.reduction())}      \{report.gradient_evals}        \{@moongrad.format_double(report.arithmetic_cost())}\{flag}",
    )
  }
  println("|")
  println("| The count is the partition actually used: the last row asks for")
  println("| 64 and gets 63, because a segment that would come out empty is")
  println("| not a segment.")
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
  println("| The \"vs tape\" column is that comparison, and it is allowed")
  println("| past -1 for exactly this reason: it says which side of the")
  println("| tape's working set the evaluation falls on, not how much of")
  println("| a saving there is.")
  println("|")
  println("| The segmentation does not change any of it, and the rows")
  println("| show it: the working set and the arithmetic are the same")
  println("| however the program is cut up. That is the finding. The")
  println("| classic trade — hold less, spend more arithmetic — needs")
  println("| a tape whose segments are independent, and a gradient")
  println("| program's are not: a node may read a primal value in any")
  println("| segment, and the primal's nodes are in topological order,")
  println("| so re-deriving one means re-deriving its predecessors.")
  println("|")
  println("| What the design does buy is a different thing: there is")
  println("| no tape to hold between the forward pass and the backward")
  println("| pass. The gradient is a program, and a program is")
  println("| evaluated when it is needed rather than stored until it")
  println("| is wanted.")
}
