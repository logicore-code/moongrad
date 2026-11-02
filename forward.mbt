// forward.mbt
//
// Pass one: constant propagation.
//
// Evaluating the graph is a left-to-right scan of `Graph::nodes`.
// No topological sort is needed because a node can only be interned
// after its children exist, so every edge points at a lower index.
// This is the property the whole design rests on, and it is why
// `moongrad` has no scheduler to get wrong.
//
// The pass also produces a *binding*: the concrete values of the
// variables and parameters. Differentiation needs those values to
// evaluate local derivatives, so they are computed once and handed
// to the reverse pass rather than recomputed.

///|
/// Concrete inputs to a graph.
pub(all) struct Binding {
  /// Value of each differentiation variable, indexed by slot.
  vars : Array[Double]
  /// Value of each parameter vector, indexed by slot.
  params : Array[Array[Double]]
} derive(Debug)

///|
/// An empty binding.
pub fn Binding::new() -> Binding {
  { vars: [], params: [], }
}

///|
/// Set the value of variable `slot`.
pub fn Binding::set_var(self : Binding, slot : Int, value : Double) -> Unit {
  while self.vars.length() <= slot {
    self.vars.push(0.0)
  }
  self.vars[slot] = value
}

///|
/// Set the value of parameter vector `slot`.
pub fn Binding::set_param(
  self : Binding,
  slot : Int,
  values : Array[Double],
) -> Unit {
  while self.params.length() <= slot {
    self.params.push([])
  }
  self.params[slot] = values
}

///|
/// Result of a forward pass.
pub(all) struct ForwardResult {
  /// Value of every slot, indexed by slot.
  values : Array[Double]
  /// Value of the node the caller asked about.
  output : Double
} derive(Debug)

///|
/// Evaluate every node of `g` under `b` and return the value of
/// `target`.
///
/// Raises `DomainError` when an operator is applied outside its
/// domain — `log` of a non-positive number, `sqrt` of a negative
/// one — and `NonFinite` when an intermediate result leaves the
/// representable range. Failing loudly here is deliberate: a NaN
/// that survives into a gradient is much harder to trace than a
/// domain error at the node that caused it.
pub fn forward(
  g : Graph,
  b : Binding,
  target : Int,
) -> ForwardResult raise ADError {
  let values = forward_values(g, b)
  if target < 0 || target >= values.length() {
    raise DanglingNode(target)
  }
  { values, output: values[target], }
}

///|
/// Evaluate every node and return the value array. This is the
/// primitive both `forward` and the recomputation path in
/// `checkpoint.mbt` are built on.
pub fn forward_values(g : Graph, b : Binding) -> Array[Double] raise ADError {
  let n = g.nodes.length()
  let values : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    values.push(0.0)
  }
  for i = 0; i < n; i = i + 1 {
    values[i] = eval_node(g, b, values, i)
  }
  values
}

///|
/// Evaluate a single slot. Children are guaranteed to be at lower
/// indices, so `values` already holds their results.
fn eval_node(
  g : Graph,
  b : Binding,
  values : Array[Double],
  id : Int,
) -> Double raise ADError {
  match g.nodes[id] {
    Node::Const(v) => v
    Node::Var(slot) =>
      if slot < 0 || slot >= b.vars.length() {
        raise UnknownVar(slot)
      } else {
        b.vars[slot]
      }
    Node::ParamGet(slot, index) => {
      if slot < 0 || slot >= b.params.length() {
        raise UnknownParam(slot)
      }
      let vec = b.params[slot]
      if index < 0 || index >= vec.length() {
        raise ParamIndexOutOfRange(slot, index, vec.length())
      }
      vec[index]
    }
    Node::Unary(op, child) => {
      let x = values[child]
      match apply_unary(op, x) {
        Some(v) => v
        None => raise DomainError("unary \{op.symbol()} applied to \{x}")
      }
    }
    Node::Binary(op, a, c) => {
      let x = values[a]
      let y = values[c]
      match apply_binary(op, x, y) {
        Some(v) => v
        None => raise DomainError("\{op.symbol()} applied to \{x} and \{y}")
      }
    }
    Node::PowBy(exponent, base) => {
      let x = values[base]
      match safe_pow(x, exponent) {
        Some(v) => v
        None => raise DomainError("x^\{exponent} at x = \{x}")
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Forward-mode cross-check
// ---------------------------------------------------------------------------

///|
/// A value carrying its own derivative with respect to one
/// direction. This is the forward-mode counterpart to the
/// reverse-mode engine, and it exists for two reasons: it gives the
/// test suite an independent implementation to check the reverse
/// pass against, and it is the cheaper of the two when a program has
/// exactly one input and produces a scalar.
///
/// `tangent` is the directional derivative along `direction`; the
/// gradient component for that direction is `directional_derivative`.
pub(all) struct Dual {
  value : Double
  tangent : Double
} derive(Debug, Eq)

///|
/// Lift a plain value with zero tangent.
pub fn Dual::constant(v : Double) -> Dual {
  { value: v, tangent: 0.0, }
}

///|
/// Lift variable `slot` from a binding, with unit tangent when
/// `direction` selects it.
pub fn Dual::from_var(b : Binding, slot : Int, direction : Int) -> Dual {
  let v = if slot >= 0 && slot < b.vars.length() { b.vars[slot] } else { 0.0 }
  { value: v, tangent: if slot == direction { 1.0 } else { 0.0 }, }
}

///|
/// Lift a parameter read with zero tangent. Parameters are handled by
/// the reverse pass; forward mode is only used to cross-check
/// gradients with respect to variables.
pub fn Dual::from_param(b : Binding, slot : Int, index : Int) -> Dual {
  if slot < 0 || slot >= b.params.length() {
    return Dual::constant(0.0)
  }
  let vec = b.params[slot]
  if index < 0 || index >= vec.length() {
    return Dual::constant(0.0)
  }
  Dual::constant(vec[index])
}

///|
/// Chain rule for a unary operator on a dual.
pub fn Dual::unary(d : Dual, op : UnaryOp) -> Dual {
  {
    value: match apply_unary(op, d.value) {
      Some(v) => v
      None => 0.0
    },
    tangent: unary_derivative(op, d.value) * d.tangent,
  }
}

///|
/// Chain rule for a binary operator on two duals.
pub fn Dual::binary(op : BinaryOp, a : Dual, b : Dual) -> Dual {
  {
    value: match apply_binary(op, a.value, b.value) {
      Some(v) => v
      None => 0.0
    },
    tangent: binary_dleft(op, a.value, b.value) * a.tangent +
    binary_dright(op, a.value, b.value) * b.tangent,
  }
}

///|
/// Chain rule for `base ^ exponent` with a constant exponent.
pub fn Dual::pow(d : Dual, exponent : Double) -> Dual {
  let v = match safe_pow(d.value, exponent) {
    Some(r) => r
    None => 0.0
  }
  {
    value: v,
    tangent: if d.value == 0.0 {
      0.0
    } else {
      exponent * @math.pow(d.value, exponent - 1.0) * d.tangent
    },
  }
}

///|
/// Evaluate `target` in forward mode along the unit direction
/// `direction`, and return `(value, directional derivative)`.
///
/// This is an independent implementation of the same derivative the
/// reverse pass computes. Any disagreement between the two is a bug
/// in one of them, which is what the test suite uses it for.
pub fn forward_mode(
  g : Graph,
  b : Binding,
  target : Int,
  direction : Int,
) -> (Double, Double) {
  let values = forward_dual(g, b, direction)
  if target < 0 || target >= values.length() {
    return (0.0, 0.0)
  }
  (values[target].value, values[target].tangent)
}

///|
/// Evaluate every node in forward mode.
pub fn forward_dual(g : Graph, b : Binding, direction : Int) -> Array[Dual] {
  let n = g.nodes.length()
  let values : Array[Dual] = []
  for i = 0; i < n; i = i + 1 {
    values.push(Dual::constant(0.0))
  }
  for i = 0; i < n; i = i + 1 {
    values[i] = dual_of_node(g, b, values, i, direction)
  }
  values
}

///|
fn dual_of_node(
  g : Graph,
  b : Binding,
  values : Array[Dual],
  id : Int,
  direction : Int,
) -> Dual {
  match g.nodes[id] {
    Node::Const(v) => Dual::constant(v)
    Node::Var(slot) => Dual::from_var(b, slot, direction)
    Node::ParamGet(slot, index) => Dual::from_param(b, slot, index)
    Node::Unary(op, child) => Dual::unary(values[child], op)
    Node::Binary(op, a, c) => Dual::binary(op, values[a], values[c])
    Node::PowBy(exponent, base) => Dual::pow(values[base], exponent)
  }
}
