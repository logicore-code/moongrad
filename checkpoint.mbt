// checkpoint.mbt
//
// Trading arithmetic for memory.
//
// A textbook reverse-mode sweep needs the forward value of every
// node, because each local derivative rule reads its operand's
// value. A tape therefore stores all of them, and the memory cost of
// one backward pass through an n-node program is O(n) — for a large
// model, the difference between fitting and swapping.
//
// This engine does not keep a tape. The backward sweep emits the
// gradient as a *program*: every adjoint is an expression over the
// primal's own nodes, and evaluating that program recomputes
// whatever it needs. The tape is gone, and with it the reason the
// sweep had to hold anything at all.
//
// The memory did not vanish, though; it moved. Evaluating the
// gradient program holds a value per node, and that is the quantity
// this file is about.
//
// WHAT THE CLASSICAL REMEDY WOULD DO, AND WHY IT DOES NOT PAY HERE
//
// The remedy is segmentation: cut the node array into segments, keep
// the value at each boundary, and rebuild a segment's interior when
// the pass enters it. That is the trade Chen et al. described for
// gradient checkpointing in 2016, and the bound that makes it worth
// doing is O(k + w) — k boundary values plus a window of width w —
// against a tape's O(n).
//
// That bound rests on an assumption this engine does not satisfy. A
// tape's interior depends on its segment's boundary and on nothing
// else, so dropping it and rebuilding it later costs one
// re-evaluation. A gradient program's nodes do not have that
// property: every local derivative is an expression over the primal's
// own nodes, so a gradient node refers back into the primal, at an
// arbitrary index. A segment's interior is therefore not independent
// of the rest of the program, and keeping only the boundaries would
// mean dropping primal values the program still needs.
//
// Rebuilding what was dropped is not one evaluation either. The
// primal's nodes are in topological order — children sit at lower
// indices — so re-deriving a primal value means re-deriving its
// predecessors, and the cost of a rebuild is the depth of the primal
// rather than the width of a segment.
//
// So segmentation does not pay for this program, and the way to say
// that without hand-waving is to measure it. This module evaluates
// the gradient program in segments, reports what the evaluation
// actually holds and actually spends, and the measurement is the
// finding: the working set and the evaluation count come out the
// same however the scan is cut up.
//
// What the design does buy is a different thing. There is no tape to
// hold between the forward pass and the backward pass, so there is
// nothing to checkpoint. The evaluation holds one value per node —
// the same order as the tape held — but only while it runs.

///|
/// What one backward pass through the gradient program cost.
///
/// Every field is a measurement of this engine's evaluation. None of
/// them is a claim about what the textbook segmented sweep could do:
/// `peak_live` is what the evaluation holds, `gradient_evals` is what
/// it spends on top of the forward pass the caller already paid for,
/// and `segments` records how the scan was partitioned.
#derive(Debug)
pub(all) struct CheckpointReport {
  /// Values a tape-based engine holds for the same backward pass: one
  /// per node of the primal.
  naive_live : Int32
  /// Values the evaluation actually holds at once.
  ///
  /// This is the whole working set — the primal's values and the
  /// gradient program's own — because a gradient node may refer to a
  /// primal value at any index, so no value can be dropped while it
  /// might still be read. The segmentation does not change it, and
  /// that is the module's finding rather than a shortcoming of the
  /// implementation.
  peak_live : Int32
  /// Node evaluations the gradient program spends on its own nodes,
  /// beyond the forward pass the caller already paid for.
  ///
  /// Nothing is recomputed here. The primal's values are the ones the
  /// forward pass produced, and the gradient program reads them
  /// where they lie; the extra evaluations are the gradient
  /// program's own arithmetic.
  gradient_evals : Int32
  /// Node evaluations in that original forward pass.
  forward_evals : Int32
  /// Segments the evaluation's scan was partitioned into.
  ///
  /// The partition says how the scan was walked. It does not change
  /// the working set or the evaluation count, because a gradient node
  /// may read a primal value in any segment.
  segments : Int32
}

///|
pub extend CheckpointReport with Debug::{to_repr}

///|
/// The working set against the naive tape: zero when the evaluation
/// holds exactly a tape's worth of values, negative when it holds
/// fewer and positive when it holds more.
///
/// The number is not confined to [-1, 1]. For this engine it is
/// negative, and it falls below -1 whenever the gradient program is
/// bigger than the primal it came from — which is the normal case,
/// because the program is bigger than the primal and refers back into
/// it, so evaluating the program holds more values than a tape would
/// rather than fewer. The ratio says which side of the tape's
/// working set the evaluation falls on, and how far; it is not a
/// saving that can be read as a fraction.
pub fn CheckpointReport::reduction(self : CheckpointReport) -> Double {
  if self.naive_live == 0 {
    return 0.0
  }
  1.0 - self.peak_live.to_double() / self.naive_live.to_double()
}

///|
/// Node evaluations as a multiple of the single forward pass the
/// caller had to make anyway.
///
/// The evaluation spends the primal's evaluations on the forward pass
/// the caller already paid for and the gradient program's own
/// evaluations on top, so the multiple is 1 + gradient_evals /
/// forward_evals.
///
/// It does not depend on the segmentation, and that is the finding
/// rather than a shortcoming of the implementation: the gradient
/// program's nodes refer back into the primal, so cutting the scan
/// into more pieces neither saves a re-evaluation nor lowers the
/// working set.
pub fn CheckpointReport::arithmetic_cost(self : CheckpointReport) -> Double {
  if self.forward_evals == 0 {
    return 1.0
  }
  1.0 + self.gradient_evals.to_double() / self.forward_evals.to_double()
}

///|
/// A contiguous run of node slots `[lo, hi)`.
#derive(Debug)
pub(all) struct Segment {
  lo : Int32
  hi : Int32
}

///|
pub extend Segment with Debug::{to_repr}

///|
/// Width of a segment.
pub fn Segment::width(self : Segment) -> Int32 {
  self.hi - self.lo
}

///|
/// Number of segments to use for a graph of `size` nodes when the
/// caller wants no live window wider than `target_width`.
///
/// Returns at least 1, so a degenerate size yields the plain tape
/// rather than a division by zero.
///
/// This is the count a segmented sweep would use for that window
/// width. Here it partitions the scan, and the partition does not
/// lower the working set, because a gradient node may read a primal
/// value in any segment.
pub fn choose_segment_count(size : Int32, target_width : Int32) -> Int32 {
  if size <= 0 || target_width <= 0 {
    return 1
  }
  let n = (size + target_width - 1) / target_width
  if n < 1 {
    1
  } else {
    n
  }
}

///|
/// Partition `[0, size)` into at most `count` contiguous segments of
/// roughly equal width.
///
/// The partition is a partition of the scan: it says how the
/// evaluation is walked, not what the evaluation holds.
pub fn segment_bounds(size : Int32, count : Int32) -> Array[Segment] {
  let out : Array[Segment] = []
  if size <= 0 || count <= 0 {
    return out
  }
  let n = if count > size { size } else { count }
  let width = (size + n - 1) / n
  let mut lo = 0
  for i = 0; i < n; i = i + 1 {
    let hi = if lo + width > size { size } else { lo + width }
    if hi > lo {
      out.push(Segment::{ lo, hi, })
      lo = hi
    }
  }
  out
}

///|
/// The peak the textbook segmented sweep would hold.
///
/// The sweep carries the boundary value of every segment it has not
/// yet entered, plus the window of the segment it is inside. With k
/// segments that is k-1 boundaries plus the widest window — and with
/// a single segment it is just the window, which is the same as the
/// naive tape.
///
/// This is the arithmetic of the technique, not of this engine. The
/// gradient program's nodes refer back into the primal at arbitrary
/// indices, so the evaluation cannot drop a primal value it may
/// still need, and it holds the whole working set instead. The
/// function is here as the number the technique would claim, to be
/// read against the report's.
pub fn peak_occupancy(bounds : Array[Segment], size : Int32) -> Int32 {
  if bounds.length() == 0 {
    return size
  }
  let mut widest = 0
  for s in bounds {
    if s.width() > widest {
      widest = s.width()
    }
  }
  bounds.length() - 1 + widest
}

///|
/// Node evaluations the textbook segmented sweep would spend
/// rebuilding segment interiors.
///
/// Each segment keeps its first node live as an entry point and
/// rebuilds the remaining `width - 1`. The last segment rebuilds
/// nothing, because the caller has already run a full forward pass
/// and those values are still to hand.
///
/// This engine spends none of them: the primal's values have to stay
/// live, because the gradient program reads them, so there is
/// nothing to rebuild. The function is here as the arithmetic the
/// technique would claim, to be read against the report's.
pub fn recomputation_cost(bounds : Array[Segment]) -> Int32 {
  let mut total = 0
  for i = 0; i < bounds.length(); i = i + 1 {
    if i < bounds.length() - 1 {
      let w = bounds[i].width()
      if w > 0 {
        total = total + w - 1
      }
    }
  }
  total
}

///|
/// Differentiate `target` with respect to every variable, evaluating
/// the gradient program in segments.
///
/// The gradient this returns is bit-identical to the one `grad`
/// produces, in every configuration. That is the property the tests
/// assert, because a memory strategy that quietly changes the answer
/// is worse than no memory strategy.
///
/// `segments` partitions the evaluation's scan into contiguous runs.
/// It changes how the scan is walked and nothing else. The gradient
/// program's nodes refer back into the primal at arbitrary indices,
/// so no partition of the scan lets the evaluation drop a value it
/// might still be asked for, and the working set and the evaluation
/// count come out the same in every configuration.
///
/// The report describes what the evaluation actually costs, which is
/// not what the technique is usually described as costing. The
/// gradient program is bigger than the primal it came from, and it
/// refers back into the primal, so the evaluation holds more values
/// than a tape would rather than fewer. The segmentation does not
/// change that, and the report does not pretend it does.
pub fn checkpointed_grad(
  g : Graph,
  b : Binding,
  target : Int32,
  segments : Int32,
) -> (Array[Double], CheckpointReport) raise ADError {
  let primal_size = g.nodes.length()
  if target < 0 || target >= primal_size {
    raise DanglingNode(target)
  }
  // The pass the caller pays for regardless of strategy.
  let values = forward_values(g, b)
  let (grads, program_size, used) = sweep(g, b, target, values, segments)
  // What the evaluation holds: one value per node of the program it
  // ran, the primal's and the gradient's alike. Nothing is dropped,
  // because a gradient node may read a primal value at any index.
  let held = program_size
  // What a tape-based engine would hold for the same backward pass:
  // one value per node of the primal.
  let naive = primal_size
  // The evaluations the gradient program adds on top of the forward
  // pass the caller already paid for.
  let extra = program_size - primal_size
  let report = CheckpointReport::{
    naive_live: naive,
    peak_live: held,
    gradient_evals: extra,
    forward_evals: primal_size,
    segments: used,
  }
  (grads, report)
}

///|
/// The backward sweep, and the evaluation of the program it emits.
///
/// The sweep itself holds nothing but the adjoint array: every local
/// derivative is an expression, so no forward value has to be read
/// here. What is left is the evaluation, and that is where the values
/// are held.
///
/// The evaluation walks the program left to right, one segment at a
/// time. The primal's values are the ones the caller's forward pass
/// produced, and they stay where they are: the gradient's own nodes
/// read them, at whatever index they happen to sit. The gradient's
/// nodes are evaluated as the scan reaches them. Nothing is evicted,
/// so the segment count changes how the scan is walked and nothing
/// else.
///
/// The adjoint array is the one thing that is genuinely O(n) and
/// genuinely has to be: every node can be a consumer of an adjoint,
/// and the sweep visits them out of order relative to arrival. In a
/// production system this is where the gradient slots would live, and
/// it is the part that scales with model size rather than with graph
/// size. The forward values, which is what this file is about, do
/// not.
fn sweep(
  g : Graph,
  b : Binding,
  target : Int32,
  values : Array[Double],
  segments : Int32,
) -> (Array[Double], Int32, Int32) raise ADError {
  let h = mirror(g)
  let zero = h.constant(0.0)
  let adj : Array[Int32] = []
  for i = 0; i < g.nodes.length(); i = i + 1 {
    adj.push(zero)
  }
  let param_adj : Array[Array[Int32]] = []
  for slot = 0; slot < g.param_shapes.length(); slot = slot + 1 {
    let row : Array[Int32] = []
    for j = 0; j < g.param_shapes[slot]; j = j + 1 {
      row.push(zero)
    }
    param_adj.push(row)
  }
  adj[target] = h.constant(1.0)
  // The sweep itself holds nothing but the adjoint array: every
  // local derivative is an expression, so no forward value has to
  // be read here.
  for i = g.nodes.length() - 1; i >= 0; i = i - 1 {
    if adj[i] == zero {
      continue
    }
    push_adjoints(g, h, adj, param_adj, i)
  }
  // What is left is the evaluation. The primal's values are already
  // in `values`, and they stay there: a gradient node may read a
  // primal value at any index, so the primal's half of the array is
  // as live at the end of the scan as it was at the start.
  let primal_size = g.nodes.length()
  let program_size = h.nodes.length()
  let bounds = segment_bounds(program_size, segments)
  while values.length() < program_size {
    values.push(0.0)
  }
  let out : Array[Double] = []
  for slot = 0; slot < g.var_count; slot = slot + 1 {
    out.push(0.0)
  }
  for s = 0; s < bounds.length(); s = s + 1 {
    let seg = bounds[s]
    for i = seg.lo; i < seg.hi; i = i + 1 {
      if i >= primal_size {
        values[i] = eval_node(h, b, values, i)
      }
    }
  }
  // The answer is the *adjoint* of each variable, not the variable.
  // The adjoint is a node of the gradient program, so its value is
  // read out of the values array after the program has been
  // evaluated.
  for slot = 0; slot < g.var_count; slot = slot + 1 {
    match g.var_slot(slot) {
      Some(node) => out[slot] = values[adj[node]]
      None => out[slot] = 0.0
    }
  }
  (out, program_size, bounds.length())
}
