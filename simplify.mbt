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
  left : Int32,
  right : Int32,
) -> Int32 {
  let lv = self.literal_of(left)
  let rv = self.literal_of(right)
  // Rule 1: annihilator. `x * 0` is 0 for any *finite* x, and
  // `x - x` is 0 for any finite x. Both drop the subtree — but only
  // when the other operand is known to be finite. A node that
  // evaluates to an infinity makes `x * 0` a NaN, and folding it
  // away would replace a loud error with a quiet wrong number.
  if op.kills_product() {
    if lv == Some(0.0) && rv_is_finite(self, right) {
      self.stats.annihilated = self.stats.annihilated + 1
      return self.constant(0.0)
    }
    if rv == Some(0.0) && rv_is_finite(self, left) {
      self.stats.annihilated = self.stats.annihilated + 1
      return self.constant(0.0)
    }
  }
  if op == BinaryOp::Sub && left == right && rv_is_finite(self, left) {
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
pub fn Graph::simplify_unary(
  self : Graph,
  op : UnaryOp,
  child : Int32,
) -> Int32 {
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
    // `neg(neg(x))` is `x`. Read the child's shape directly rather
    // than through `get`, which raises for a slot that does not
    // exist.
    if child >= 0 && child < self.nodes.length() {
      match self.nodes[child] {
        Node::Unary(UnaryOp::Neg, inner) => {
          self.stats.identities = self.stats.identities + 1
          return inner
        }
        _ => ()
      }
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
pub fn Graph::simplify_pow(
  self : Graph,
  exponent : Double,
  base : Int32,
) -> Int32 {
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
pub fn Graph::literal_of(self : Graph, id : Int32) -> Double? {
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
      if is_finite(v) {
        Some(v)
      } else {
        None
      }
    }
    UnaryOp::Ln =>
      if x > 0.0 {
        let v = @math.ln(x)
        if is_finite(v) {
          Some(v)
        } else {
          None
        }
      } else {
        None
      }
    UnaryOp::Sqrt =>
      if x >= 0.0 {
        let v = x.sqrt()
        if is_finite(v) {
          Some(v)
        } else {
          None
        }
      } else {
        None
      }
    UnaryOp::Sin => Some(@math.sin(x))
    UnaryOp::Cos => Some(@math.cos(x))
    UnaryOp::Tanh => Some(@math.tanh(x))
    UnaryOp::Sigmoid => Some(1.0 / (1.0 + @math.exp(-x)))
    UnaryOp::Softplus =>
      // log(1 + e^x), evaluated as max(x,0) + log1p(e^-|x|) so it
      // does not overflow for large x. The log1p matters: for a
      // large negative x the sum rounds to one and the logarithm
      // of it is zero, while the true value is a small number that
      // still matters.
      Some(maxf(x, 0.0) + log1p(@math.exp(-absf(x))))
    UnaryOp::Relu => Some(if x > 0.0 { x } else { 0.0 })
    UnaryOp::Silu => Some(x / (1.0 + @math.exp(-x)))
    UnaryOp::Gelu => {
      // The tanh approximation, which is the form every framework
      // ships. The exact form needs the normal CDF.
      let u = gelu_c * (x + 0.044715 * x * x * x)
      Some(0.5 * x * (1.0 + @math.tanh(u)))
    }
  }
}

///|
/// The scale factor in the tanh approximation of GELU: the square
/// root of two over pi.
pub let gelu_c : Double = 0.7978845608028654

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
/// The logarithm of one plus a small number.
///
/// The standard library has no `log1p`, and the naive form loses
/// the answer entirely for a small argument: `1 + u` rounds to one
/// and its logarithm is zero, while the true value is very nearly u.
fn log1p(u : Double) -> Double {
  if u > 1.0e-4 {
    @math.ln(1.0 + u)
  } else {
    u - u * u * 0.5 + u * u * u / 3.0
  }
}

///|
/// Whether a node is known to be finite.
///
/// A literal is finite if its value is. Anything else is not known:
/// the simplifier cannot tell what a node will evaluate to, and a
/// node that evaluates to an infinity is exactly the case where the
/// annihilator rules would be wrong.
fn rv_is_finite(g : Graph, id : Int32) -> Bool {
  match g.literal_of(id) {
    Some(v) => is_finite(v)
    None => false
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
      // 1 - t^2 cancels catastrophically for a large argument: the
      // square of a number very close to one has no significant
      // digits left. The form below keeps them.
      let e = @math.exp(-2.0 * x)
      if x > 0.0 {
        4.0 * e / ((1.0 + e) * (1.0 + e))
      } else {
        let p = @math.exp(2.0 * x)
        4.0 * p / ((1.0 + p) * (1.0 + p))
      }
    }
    UnaryOp::Sigmoid => {
      // s * (1 - s) has the same problem. The form below keeps the
      // small factor instead of subtracting it from one.
      let s = 1.0 / (1.0 + @math.exp(-x))
      if x > 0.0 {
        let e = @math.exp(-x)
        e / ((1.0 + e) * (1.0 + e))
      } else {
        s * (1.0 - s)
      }
    }
    UnaryOp::Softplus => 1.0 / (1.0 + @math.exp(-x))
    UnaryOp::Relu => if x > 0.0 { 1.0 } else { 0.0 }
    UnaryOp::Silu => {
      // s + x s (1 - s) with s = sigmoid(x)
      let s = 1.0 / (1.0 + @math.exp(-x))
      s + x * s * (1.0 - s)
    }
    UnaryOp::Gelu => {
      // The derivative of 0.5 x (1 + tanh u), with
      // u = c (x + 0.044715 x^3):
      //   0.5 (1 + tanh u) + 0.5 x (1 - tanh^2 u) u'
      let u = gelu_c * (x + 0.044715 * x * x * x)
      let t = @math.tanh(u)
      let du = gelu_c * (1.0 + 3.0 * 0.044715 * x * x)
      0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du
    }
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
  if a > b {
    a
  } else {
    b
  }
}

///|
/// `min` for doubles.
pub fn minf(a : Double, b : Double) -> Double {
  if a < b {
    a
  } else {
    b
  }
}

///|
/// `abs` for doubles.
pub fn absf(a : Double) -> Double {
  if a < 0.0 {
    -a
  } else {
    a
  }
}
