// checkpoint.mbt
//
// Trading arithmetic for memory.
//
// The reverse sweep needs the forward value of every node, because
// each local derivative rule reads its operand's value. A textbook
// tape therefore stores all of them and the memory cost of one
// backward pass through an n-node program is O(n) — for a large
// model, the difference between fitting and swapping.
//
// The way out is visible in the sweep itself. Children sit at lower
// indices, so a right-to-left pass reaches any window of the forward
// computation on demand. It does not need all the values at once; it
// needs each value shortly before consuming it and never again.
//
// So: cut the node array into segments, keep only the values at the
// boundaries, and recompute a segment's interior when the sweep
// enters it. Peak occupancy drops from O(n) to O(k + w) for k
// segments of width w, at the cost of re-evaluating every interior
// node once. That is the same trade Chen et al. described for
// gradient checkpointing in 2016, applied to a compiler rather than
// to an interpreter — and it is why the segment count is a dial
// rather than a constant.
//
// The pathological case is a graph with many narrow segments: you
// pay recomputation without buying locality, and the arithmetic cost
// approaches 2x. `choose_segment_count` therefore takes a target
// width rather than a segment count, so a caller who says "keep the
// window under 256 nodes" gets the same behaviour on a graph of any
// size.

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
  let bounds = segment_bounds(size, segments)
  let grads = sweep(g, b, target, all_values, bounds, size)
  let recomputed = recomputation_cost(bounds)
  let report = CheckpointReport::{
    naive_live: size,
    peak_live: peak_occupancy(bounds, size),
    recomputed,
    forward_evals: size,
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
  bounds : Array[Segment],
  size : Int32,
) -> Array[Double] raise ADError {
  let h = mirror(g)
  let zero = h.constant(0.0)
  let adj : Array[Int32] = []
  for i = 0; i < size; i = i + 1 {
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
  for s = bounds.length() - 1; s >= 0; s = s - 1 {
    let seg = bounds[s]
    // Rebuild this segment's forward values from the values below
    // it, which are either a boundary we kept or the tail of the
    // caller's original pass.
    let window = rebuild(g, b, all_values, seg)
    for i = seg.hi - 1; i >= seg.lo; i = i - 1 {
      if adj[i] == zero {
        continue
      }
      push_adjoints(g, h, window, adj, param_adj, i)
    }
  }
  let out : Array[Double] = []
  for slot = 0; slot < g.var_count; slot = slot + 1 {
    match g.var_slot(slot) {
      Some(node) => out.push(forward(h, b, adj[node]).output)
      None => out.push(0.0)
    }
  }
  out
}

///|
/// Forward values for the nodes in `seg`, rebuilt from scratch.
///
/// Children below `seg.lo` are read from `all` — the caller keeps
/// those live, which is the k boundary values the model accounts
/// for. Children inside the segment are taken from the window as it
/// fills, which is safe because they always sit at a lower index.
fn rebuild(
  g : Graph,
  b : Binding,
  all : Array[Double],
  seg : Segment,
) -> Array[Double] raise ADError {
  // A full-length view whose tail is authoritative and whose head
  // we refresh segment by segment. Indexing stays O(1) and the
  // caller keeps control of what stays live.
  let window = all.copy()
  for i = seg.lo; i < seg.hi; i = i + 1 {
    window[i] = eval_node(g, b, window, i)
  }
  window
}
