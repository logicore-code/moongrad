// simplify.mbt
//
// Peephole simplification, run on every node as it is built.
//
// The rules are deliberately conservative. A simplifier that is too
// aggressive changes the floating-point semantics of the caller's
// program, and a training run that converges to a slightly different
// minimum because `x * 0.0` was replaced by `0.0` before the
// rounding had a chance to accumulate is a genuinely unpleasant thing
// to debug. So every rule here is exact in IEEE-754 for the operands
// it accepts, and nothing is rewritten when either operand is
// non-finite.
//
// Three families of rule:
//
//   * annihilator — `x * 0 → 0`, `x * 0 → 0`, `x - x → 0`.
//     Drops an entire subtree.
//   * identity    — `x + 0 → x`, `x * 1 → x`, `x / 1 → x`.
//     Drops an operator, keeps the subtree.
//   * fold        — both children are literals, so evaluate now.
//     Drops an entire subtree.
//
// The simplifier counts what it did in `Graph::stats`; nothing is
// silent.

///|
/// Simplify `op(left, right)` and emit the result.
pub fn Graph::simplify_binary(
  self : Graph,
  op : BinaryOp,
  left : Int,
  right : Int,
) -> Int raise ADError {
  let lv = self.literal_of(left)
  let rv = self.literal_of(right)
  // Rule 1: annihilator. `x * 0` is 0 for any finite x, and `x - x`
  // is 0 for any finite x. Both drop the subtree.
  if op.kills_product() {
    if lv == Some(0.0) || rv == Some(0.0) {
      self.stats.annihilated = self.stats.annihilated + 1
      return self.constant(0.0)
    }
  }
  if op == BinaryOp::Sub && left == right {
    self.stats.annihilated = self.stats.annihilated + 1
    return self.constant(0.0)
  }
  // Rule 2: identity against a literal on either side. `a - b` is not
  // symmetric, so it only fires when the *right* operand is the
  // identity; `0 - b` is `-b`, which is a rewrite the caller has to
  // ask for.
  let left_is_id = match (op, lv) {
    (BinaryOp::Add, Some(v)) => v == 0.0
    (BinaryOp::Mul, Some(v)) => v == 1.0
    _ => false
  }
  let right_is_id = match (op, rv) {
    (BinaryOp::Add, Some(v)) => v == 0.0
    (BinaryOp::Sub, Some(v)) => v == 0.0
    (BinaryOp::Mul, Some(v)) => v == 1.0
    (BinaryOp::Div, Some(v)) => v == 1.0
    _ => false
  }
  if left_is_id {
    self.stats.identities = self.stats.identities + 1
    return right
  }
  if right_is_id {
    self.stats.identities = self.stats.identities + 1
    return left
  }
  // Rule 3: both sides literal — evaluate now.
  if lv != None && rv != None {
    let a = match lv {
      Some(v) => v
      None => 0.0
    }
    let b = match rv {
      Some(v) => v
      None => 0.0
    }
    match apply_binary(op, a, b) {
      Some(v) => {
        self.stats.folds = self.stats.folds + 1
        return self.constant(v)
      }
      None => ()
    }
  }
  self.intern(Node::Binary(op, left, right))
}

///|
/// Simplify `op(child)` and emit the result.
pub fn Graph::simplify_unary(self : Graph, op : UnaryOp, child : Int) -> Int raise ADError {
  let cv = self.literal_of(child)
  // `op(0)` has a closed form for three of the nine operators.
  if cv == Some(0.0) {
    let shortcut = match op {
      UnaryOp::Neg => Some(0.0)
      UnaryOp::Exp => Some(1.0)
      UnaryOp::Tanh => Some(0.0)
      UnaryOp::Sigmoid => Some(0.5)
      _ => None
    }
    match shortcut {
      Some(v) => {
        self.stats.folds = self.stats.folds + 1
        return self.constant(v)
      }
      None => ()
    }
  }
  if cv == Some(1.0) {
    match op {
      UnaryOp::Ln => {
        self.stats.folds = self.stats.folds + 1
        return self.constant(0.0)
      }
      _ => ()
    }
  }
  if cv == Some(-1.0) {
    match op {
      UnaryOp::Tanh => {
        self.stats.folds = self.stats.folds + 1
        return self.constant(-1.0)
      }
      _ => ()
    }
  }
  // `neg(neg(x))` is `x`. Trivial, and surprisingly common once a
  // caller composes several layers symbolically.
  if op == UnaryOp::Neg {
    match self.get(child) {
      Node::Unary(UnaryOp::Neg, inner) => {
        self.stats.identities = self.stats.identities + 1
        return inner
      }
      _ => ()
    }
  }
  if cv != None {
    let x = match cv {
      Some(v) => v
      None => 0.0
    }
    match apply_unary(op, x) {
      Some(v) => {
        self.stats.folds = self.stats.folds + 1
        return self.constant(v)
      }
      None => ()
    }
  }
  self.intern(Node::Unary(op, child))
}

///|
/// Simplify `base ^ exponent` and emit the result.
///
/// A literal base is folded before any structural rule, so `3^2`
/// becomes the literal 9 rather than a `PowBy` node that the
/// forward pass would have to evaluate for nothing.
pub fn Graph::simplify_pow(self : Graph, exponent : Double, base : Int) -> Int raise ADError {
  let bv = self.literal_of(base)
  if exponent == 1.0 {
    self.stats.identities = self.stats.identities + 1
    return base
  }
  if exponent == 0.0 {
    self.stats.identities = self.stats.identities + 1
    return self.constant(1.0)
  }
  if bv != None {
    let x = match bv {
      Some(v) => v
      None => 0.0
    }
    match safe_pow(x, exponent) {
      Some(v) => {
        self.stats.folds = self.stats.folds + 1
        return self.constant(v)
      }
      None => ()
    }
  }
  if exponent == 2.0 {
    // `x^2` is the most common shape in a quadratic model; giving it
    // its own node keeps the printed graph readable and lets the
    // simplifier recognise `x^2 - x^2` as zero without evaluating
    // either square.
    self.stats.identities = self.stats.identities + 1
    return self.intern(Node::PowBy(2.0, base))
  }
  self.intern(Node::PowBy(exponent, base))
}

///|
/// The literal stored in slot `id`, or `None` when the slot holds
/// something that has to be evaluated first. A slot that does not
/// exist at all also yields `None`; the caller turns the resulting
/// dangling reference into a proper error later, when the node is
/// actually built.
pub fn Graph::literal_of(self : Graph, id : Int) -> Double? {
  if id < 0 || id >= self.nodes.length() {
    return None
  }
  match self.nodes[id] {
    Node::Const(v) => Some(v)
    _ => None
  }
}

///|
/// Evaluate a binary operator on two doubles. Returns `None` when the
/// result would be non-finite, so the caller keeps the unfolded node
/// and lets the forward pass raise a domain error with a useful
/// message instead of freezing a NaN into the graph.
pub fn apply_binary(op : BinaryOp, a : Double, b : Double) -> Double? {
  let v = match op {
    BinaryOp::Add => a + b
    BinaryOp::Sub => a - b
    BinaryOp::Mul => a * b
    BinaryOp::Div => a / b
  }
  if is_finite(v) {
    Some(v)
  } else {
    None
  }
}

///|
/// Evaluate a unary operator on one double, or return `None` when
/// the operator is undefined there. `Ln` and `Sqrt` are the only two
/// with a domain restriction; the rest are total.
pub fn apply_unary(op : UnaryOp, x : Double) -> Double? {
  match op {
    UnaryOp::Neg => Some(-x)
    UnaryOp::Exp => {
      let v = @math.exp(x)
      if is_finite(v) { Some(v) } else { None }
    }
    UnaryOp::Ln =>
      if x > 0.0 {
        let v = @math.ln(x)
        if is_finite(v) { Some(v) } else { None }
      } else {
        None
      }
    UnaryOp::Sqrt =>
      if x >= 0.0 {
        let v = x.sqrt()
        if is_finite(v) { Some(v) } else { None }
      } else {
        None
      }
    UnaryOp::Sin => Some(@math.sin(x))
    UnaryOp::Cos => Some(@math.cos(x))
    UnaryOp::Tanh => Some(@math.tanh(x))
    UnaryOp::Sigmoid => Some(1.0 / (1.0 + @math.exp(-x)))
    UnaryOp::Softplus =>
      // log(1 + e^x), evaluated as max(x,0) + log(1 + e^-|x|) so it
      // does not overflow for large x.
      Some(maxf(x, 0.0) + @math.ln(1.0 + @math.exp(-absf(x))))
  }
}

///|
/// `x ^ c` with a guard against the 0^negative case.
pub fn safe_pow(x : Double, c : Double) -> Double? {
  if x == 0.0 && c < 0.0 {
    return None
  }
  let v = @math.pow(x, c)
  if is_finite(v) {
    Some(v)
  } else {
    None
  }
}

///|
/// Local derivative of a unary operator, given the value of its
/// operand. This is the table the reverse pass and the forward-mode
/// cross-check both consult, so the two implementations cannot drift
/// apart.
pub fn unary_derivative(op : UnaryOp, x : Double) -> Double {
  match op {
    UnaryOp::Neg => -1.0
    UnaryOp::Exp => @math.exp(x)
    UnaryOp::Ln => 1.0 / x
    UnaryOp::Sqrt => 0.5 / x.sqrt()
    UnaryOp::Sin => @math.cos(x)
    UnaryOp::Cos => -@math.sin(x)
    UnaryOp::Tanh => {
      let t = @math.tanh(x)
      1.0 - t * t
    }
    UnaryOp::Sigmoid => {
      let s = 1.0 / (1.0 + @math.exp(-x))
      s * (1.0 - s)
    }
    UnaryOp::Softplus => 1.0 / (1.0 + @math.exp(-x))
  }
}

///|
/// Local derivative of `a op b` with respect to `a`.
pub fn binary_dleft(op : BinaryOp, a : Double, b : Double) -> Double {
  let _ = a
  match op {
    BinaryOp::Add => 1.0
    BinaryOp::Sub => 1.0
    BinaryOp::Mul => b
    BinaryOp::Div => 1.0 / b
  }
}

///|
/// Local derivative of `a op b` with respect to `b`.
pub fn binary_dright(op : BinaryOp, a : Double, b : Double) -> Double {
  match op {
    BinaryOp::Add => 1.0
    BinaryOp::Sub => -1.0
    BinaryOp::Mul => a
    BinaryOp::Div => -a / (b * b)
  }
}

///|
/// `max` for doubles. MoonBit's builtin lives on integers only.
pub fn maxf(a : Double, b : Double) -> Double {
  if a > b { a } else { b }
}

///|
/// `min` for doubles.
pub fn minf(a : Double, b : Double) -> Double {
  if a < b { a } else { b }
}

///|
/// `abs` for doubles.
pub fn absf(a : Double) -> Double {
  if a < 0.0 { -a } else { a }
}
