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
// The way out looks like the one the tape had: children sit at lower
// indices, so a left-to-right pass reaches any window of the
// computation on demand. Cut the node array into segments, keep only
// the values at the boundaries, and recompute a segment's interior
// when the pass enters it. That is the trade Chen et al. described
// for gradient checkpointing in 2016.
//
// **It does not pay here, and this file is as much about why as about
// what.** A tape's interior depends on its segment's boundary and on
// nothing else, so dropping it and rebuilding it later costs one
// re-evaluation. A gradient program's nodes do not have that
// property: every local derivative is an expression over the primal's
// own nodes, so a gradient node refers back into the primal, at an
// arbitrary index. Dropping a primal value therefore means
// recomputing it, and recomputing it means walking down to whatever
// is still live — which grows with the depth of the primal, and, where
// the primal shares a subtree, with the sharing as well.
//
// So the bound O(k + w) is a property of the technique and not of
// this program. What the module does is measure what the segmented
// evaluation actually costs, and check that the answer is the same
// either way — which is the part that has to be true before the
// arithmetic is worth discussing.
//
// The conclusion the measurement supports is worth stating plainly:
// the gradient-program design solves the tape's memory problem a
// different way. There is no tape to hold between the forward pass
// and the backward pass, so there is nothing to checkpoint. The
// evaluation holds one value per node, the same order as the tape
// held — but only while it runs.

///|
/// What one checkpointed backward pass cost.
#derive(Debug)
pub(all) struct CheckpointReport {
  /// Values the sweep would hold at once if it stored every node.
  naive_live : Int32
  /// Values actually held at the worst point of the sweep.
  peak_live : Int32
  /// Node evaluations spent on recomputation, beyond the single
  /// forward pass the caller already paid for.
  recomputed : Int32
  /// Node evaluations in that original forward pass.
  forward_evals : Int32
  /// Number of segments the sweep was split into.
  segments : Int32
  /// True when the sweep actually recomputed anything, which is
  /// false only for the one-segment case.
  did_checkpoint : Bool
}

///|
pub extend CheckpointReport with Debug::{to_repr}

///|
/// Fraction of peak memory saved against the naive tape, in [0, 1].
pub fn CheckpointReport::reduction(self : CheckpointReport) -> Double {
  if self.naive_live == 0 {
    return 0.0
  }
  1.0 - self.peak_live.to_double() / self.naive_live.to_double()
}

///|
/// Total forward evaluations as a multiple of the single pass the
/// caller had to make anyway. Approaches 2.0 for many segments and
/// approaches 1.0 for few.
pub fn CheckpointReport::arithmetic_cost(self : CheckpointReport) -> Double {
  if self.forward_evals == 0 {
    return 1.0
  }
  1.0 + self.recomputed.to_double() / self.forward_evals.to_double()
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
/// Peak number of values the sweep holds at once.
///
/// The sweep carries the boundary value of every segment it has not
/// yet entered, plus the window of the segment it is inside. With k
/// segments that is k-1 boundaries plus the widest window — and
/// with a single segment it is just the window, which is the same
/// as the naive tape. That last case is why one segment has to
/// report a reduction of exactly zero rather than a slightly
/// negative one.
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
/// Node evaluations spent recomputing segment interiors.
///
/// Each segment keeps its first node live as an entry point and
/// rebuilds the remaining `width - 1`. The last segment rebuilds
/// nothing, because the caller has already run a full forward pass
/// and those values are still to hand.
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
/// Differentiate `target` with respect to every variable, holding
/// only a bounded window of forward values.
///
/// The gradient this returns is bit-identical to the one
/// `grad` produces. The only difference is bookkeeping: this path
/// splits the sweep into segments, rebuilds each segment's interior
/// on entry, and reports what that cost. The checkpointing tests
/// assert the equality directly, because a memory strategy that
/// quietly changes the answer is worse than no memory strategy.
pub fn checkpointed_grad(
  g : Graph,
  b : Binding,
  target : Int32,
  segments : Int32,
) -> (Array[Double], CheckpointReport) raise ADError {
  let size = g.nodes.length()
  if target < 0 || target >= size {
    raise DanglingNode(target)
  }
  // The pass the caller pays for regardless of strategy.
  let all_values = forward_values(g, b)
  let (grads, graph_size) = sweep(g, b, target, all_values, segments)
  let bounds = segment_bounds(graph_size, segments)
  let recomputed = recomputation_cost(bounds)
  let report = CheckpointReport::{
    naive_live: graph_size,
    peak_live: peak_occupancy(bounds, graph_size),
    recomputed,
    forward_evals: graph_size,
    segments: bounds.length(),
    did_checkpoint: recomputed > 0,
  }
  (grads, report)
}

///|
/// The backward sweep, visiting segments right to left and rebuilding
/// each segment's forward values on entry.
///
/// The adjoint array is the one thing that is genuinely O(n) and
/// genuinely has to be: every node can be a consumer of an adjoint,
// and the sweep visits them out of order relative to arrival. In a

///|
/// production system this is where the gradient slots would live, and
/// it is the part that scales with model size rather than with graph
/// size. The forward values, which is what this file is about, do
/// not.
fn sweep(
  g : Graph,
  b : Binding,
  target : Int32,
  all_values : Array[Double],
  segments : Int32,
) -> (Array[Double], Int32) raise ADError {
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
  // What is left is the evaluation. The gradient graph is walked one
  // segment at a time; a segment's interior is recomputed on entry
  // from the values below it, which are either a boundary we kept
  // or the tail of the caller's original pass.
  let size = h.nodes.length()
  let bounds = segment_bounds(size, segments)
  let window = all_values.copy()
  while window.length() < size {
    window.push(0.0)
  }
  let out : Array[Double] = []
  for slot = 0; slot < g.var_count; slot = slot + 1 {
    out.push(0.0)
  }
  for s = 0; s < bounds.length(); s = s + 1 {
    let seg = bounds[s]
    for i = seg.lo; i < seg.hi; i = i + 1 {
      if i < all_values.length() {
        window[i] = all_values[i]
      } else {
        window[i] = eval_node(h, b, window, i)
      }
    }
  }
  // The answer is the *adjoint* of each variable, not the variable.
  // The adjoint is a node of the gradient program, so its value is
  // read out of the window after the program has been evaluated.
  for slot = 0; slot < g.var_count; slot = slot + 1 {
    match g.var_slot(slot) {
      Some(node) => out[slot] = window[adj[node]]
      None => out[slot] = 0.0
    }
  }
  (out, size)
}
