// tensor_simplify.mbt
//
// Peephole simplification and node sharing for the tensor graph.
//
// The rules are the ones simplify.mbt applies to the scalar graph, and
// the two engines ask the *same functions* whether a literal is an
// identity, so the tables cannot drift apart.
//
// Why the tensor engine needs its own: the backward sweep expands
// every local derivative rule into arithmetic, and that arithmetic is
// where the node count multiplies. A gradient program for a small
// network came out several times the size of the primal it came from,
// with the same subexpression written out again and again — `x + 0`
// where `x` is the adjoint accumulator, `g * s` where `g` is the
// incoming gradient. The scalar engine has always simplified as it
// builds; the tensor engine did not, and that was the largest
// avoidable cost in the gradient program.
//
// Two families of rule are applied:
//
//   * identity — `x + 0`, `x * 1`, `x - 0`, `x / 1` and their
//     mirror images, where the literal is a *scalar*, so the shape of
//     the result is the shape of the operand that survives.
//   * fold — both operands are literals, so the operation is
//     performed now and the subtree disappears.
//
// The scalar engine's annihilators (`x * 0` and `x - x`) are
// deliberately not applied here. A scalar has no shape and a tensor
// does: `x * 0` is a zero *of x's shape*, not a zero scalar, so
// folding it would change the shape of the result and every operation
// downstream of it. Leaving those rules to the scalar side is the
// conservative choice, and the tensor engine has no need for them:
// its adjoints are accumulated by addition, not by multiplication.

///|
/// The value of a node, if the node is a literal scalar.
///
/// A literal scalar is a constant whose shape is empty. A constant
/// of any other shape is not an identity for anything: `x + [0, 0]`
/// is not `x`, it is `x` with a zero row added.
fn t_scalar_literal(g : TGraph, id : Int32) -> Double? {
  if id < 0 || id >= g.nodes.length() {
    return None
  }
  match g.nodes[id] {
    TNode::Const(slot) => {
      if slot < 0 || slot >= g.pool.length() {
        return None
      }
      let t = g.pool[slot]
      if t.shape.length() == 0 && t.data.length() == 1 {
        Some(t.data[0])
      } else {
        None
      }
    }
    _ => None
  }
}

///|
/// The tensor of a node, if the node is a constant.
fn t_tensor_literal(g : TGraph, id : Int32) -> Tensor? {
  if id < 0 || id >= g.nodes.length() {
    return None
  }
  match g.nodes[id] {
    TNode::Const(slot) =>
      if slot < 0 || slot >= g.pool.length() {
        None
      } else {
        Some(g.pool[slot])
      }
    _ => None
  }
}

///|
/// Whether every element of `t` is a finite number.
///
/// A fold that produced an infinity or a NaN is not a fold.
///
/// The reason is not that the fold would hide an error — the tensor
/// engine does not raise on a division by zero, it produces an
/// infinity and carries on — but that a non-finite constant is a
/// value the caller cannot undo. A parameter that becomes infinite
/// stays infinite; folding it in early would put it there before the
/// caller had a chance to notice that the gradient was already
/// diverged.
fn t_all_finite(t : Tensor) -> Bool {
  for v in t.data {
    if v != v {
      return false
    }
    if v > 1.7976931348623157e308 || v < -1.7976931348623157e308 {
      return false
    }
  }
  true
}

///|
/// Build `a op b`, simplifying where it is safe to do so.
///
/// The identity rules are the scalar engine's, read out of
/// `simplify.mbt`; the fold is performed on tensors rather than on
/// doubles.
pub fn TGraph::simplify_binary(
  self : TGraph,
  op : BinaryOp,
  a : Int32,
  b : Int32,
) -> Int32 {
  self.stats.requests = self.stats.requests + 1
  // Identity on the left. `0 + b` is `b` and `1 * b` is `b`, and the
  // shape of the result is the shape of `b` either way, because the
  // literal is a scalar and broadcasts to whatever `b` is.
  match t_scalar_literal(self, a) {
    Some(v) =>
      if binary_left_identity(op, v) {
        self.stats.identities = self.stats.identities + 1
        return b
      }
    None => ()
  }
  // Identity on the right. `a - 0` is `a` and `a * 1` is `a`.
  match t_scalar_literal(self, b) {
    Some(v) =>
      if binary_right_identity(op, v) {
        self.stats.identities = self.stats.identities + 1
        return a
      }
    None => ()
  }
  // Both operands are literals: perform the operation now.
  match (t_tensor_literal(self, a), t_tensor_literal(self, b)) {
    (Some(x), Some(y)) =>
      match t_binary(op, x, y) {
        Some(v) =>
          if t_all_finite(v) {
            self.stats.folds = self.stats.folds + 1
            return self.constant(v)
          }
        None => ()
      }
    _ => ()
  }
  self.intern(TNode::Binary(op, a, b))
}

///|
/// Build `op(x)`, simplifying where it is safe to do so.
pub fn TGraph::simplify_unary(self : TGraph, op : UnaryOp, x : Int32) -> Int32 {
  self.stats.requests = self.stats.requests + 1
  match t_scalar_literal(self, x) {
    Some(v) => {
      // The closed forms for a literal operand. Every one of them
      // produces a scalar, so the shape is unchanged.
      let shortcut = match op {
        UnaryOp::Neg => Some(-v)
        UnaryOp::Exp => Some(@math.exp(v))
        UnaryOp::Tanh => Some(@math.tanh(v))
        UnaryOp::Sigmoid => Some(1.0 / (1.0 + @math.exp(-v)))
        UnaryOp::Ln => if v > 0.0 { Some(@math.ln(v)) } else { None }
        UnaryOp::Sqrt => if v >= 0.0 { Some(v.sqrt()) } else { None }
        _ => None
      }
      match shortcut {
        Some(r) =>
          if r == r &&
            r <= 1.7976931348623157e308 &&
            r >= -1.7976931348623157e308 {
            self.stats.folds = self.stats.folds + 1
            return self.constant(Tensor::scalar(r))
          }
        None => ()
      }
    }
    None => ()
  }
  // `neg(neg(x))` is `x`, and it shows up whenever a caller composes
  // layers symbolically.
  if op == UnaryOp::Neg {
    if x >= 0 && x < self.nodes.length() {
      match self.nodes[x] {
        TNode::Unary(UnaryOp::Neg, inner) => {
          self.stats.identities = self.stats.identities + 1
          return inner
        }
        _ => ()
      }
    }
  }
  self.intern(TNode::Unary(op, x))
}

///|
/// Find the slot that already holds this node, or give it a new one.
///
/// Sharing is what keeps a gradient program from repeating itself.
/// The backward sweep emits `g * s` once per path that reaches a
/// softmax, and those paths are often the same computation written
/// out twice.
pub fn TGraph::intern(self : TGraph, node : TNode) -> Int32 {
  match self.intern_table.get(node) {
    Some(id) => {
      self.stats.cse_hits = self.stats.cse_hits + 1
      id
    }
    None => {
      let id = self.nodes.length()
      self.nodes.push(node)
      self.stats.nodes_built = self.stats.nodes_built + 1
      self.intern_table.set(node, id)
      id
    }
  }
}
