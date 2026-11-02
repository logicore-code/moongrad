// reverse.mbt
//
// Pass two: the gradient compiler.
//
// This is where moongrad parts company with a textbook tape-based
// autodiffer. Instead of pushing numeric adjoints into a flat array
// and reading the answer off the end, the backward sweep *emits a
// second program* — the gradient program — whose nodes are
// `d(f) / d(primal_slot)`. Two things follow from that choice.
//
// The gradient becomes a first-class value. A caller can print it,
// count its nodes, re-evaluate it at fresh inputs, or hand it back
// to the same compiler to get a second derivative. `hessian.mbt` does
// precisely that, and it is short for exactly this reason.
//
// Memory stops being a property of the tape. The adjoint of a node
// depends only on its children, and children sit at lower indices, so
// a right-to-left sweep can recompute any window of the forward pass
// instead of keeping it resident. `checkpoint.mbt` builds on that.
//
// The sweep, concretely:
//
//   adj[target] := Const(1)
//   for i := n-1 downto 0:
//       if adj[i] is not the zero constant:
//           emit adj[i] * d(nodes[i])/d(child) into adj[child] for
//           every child, appending the product to the graph
//
// Emitted nodes always land at indices above `i`, so the sweep never
// revisits work it has already done and never needs a worklist.
//
// One detail is worth dwelling on, because it is the decision that
// separates this from a tape. A tape-based engine evaluates each
// local derivative at the current input and stores the resulting
// *number*. Here the rules emit the derivative *expression* instead:
// `d/dx sin(x)` becomes a `cos` node applied to the primal's `x`
// node, not the number `cos(1.7)`.
//
// Freezing numbers would have been simpler and wrong. The gradient
// program has to be a function of its inputs, because the caller
// will want to evaluate it somewhere else — the next step of a
// gradient descent, the interior of a line search, or a second
// derivative — and a frozen gradient silently returns the answer
// for the point it was compiled at. Emitting the expression costs a
// few more nodes and makes the program re-evaluable, which in turn
// is what lets `hessian.mbt` be eleven lines rather than a second
// engine.

///|
/// Output of one symbolic backward sweep.
pub(all) struct GradGraph {
  /// The graph the sweep emitted into. Its first `primal_size` nodes
  /// are structurally identical to the primal's; everything above is
  /// gradient arithmetic.
  graph : Graph
  /// Number of leading nodes that came from the primal.
  primal_size : Int
  /// Slot in `graph` of `d(target) / d(Var(i))`, one per variable.
  var_adj : Array[Int]
  /// Slot in `graph` of the accumulated `d(target) / d(param)`,
  /// indexed by parameter slot then by element index. A node rather
  /// than a number, for the same reason `var_adj` is a node: the
  /// gradient has to be re-evaluable, and a parameter gradient that
  /// is frozen at the compile point is as wrong as a variable
  /// gradient that is.
  param_adj_node : Array[Array[Int]]
} derive(Debug)

///|
/// A primal graph paired with its compiled gradient.
pub(all) struct Program {
  primal : Graph
  grad : GradGraph
  /// Value of every primal slot under the binding the gradient was
  /// compiled against.
  primal_values : Array[Double]
  /// Value of the differentiated node.
  output : Double
} derive(Debug)

///|
/// Compile the gradient of `target` with respect to every variable
/// and every parameter.
///
/// The primal values must already be available; the backward rules
/// need them to turn local derivatives into constants. Pass the
/// output of `forward_values`.
///
/// The returned graph is a fresh object, so mutating the primal
/// afterwards cannot corrupt a gradient that has already been
/// compiled.
pub fn compile_gradient(
  g : Graph,
  b : Binding,
  primal_values : Array[Double],
  target : Int,
) -> GradGraph raise ADError {
  if target < 0 || target >= g.nodes.length() {
    raise DanglingNode(target)
  }
  let h = mirror(g)
  let n = g.nodes.length()
  // Adjoints, one slot per primal node, each holding a node in `h`
  // that evaluates to the partial derivative. `zero` is the shared
  // literal so the simplifier collapses the common case.
  let zero = h.constant(0.0)
  let adj : Array[Int] = []
  for i = 0; i < n; i = i + 1 {
    adj.push(zero)
  }
  let param_adj : Array[Array[Int]] = []
  for slot = 0; slot < g.param_shapes.length(); slot = slot + 1 {
    let row : Array[Int] = []
    for i = 0; i < g.param_shapes[slot]; i = i + 1 {
      row.push(zero)
    }
    param_adj.push(row)
  }
  adj[target] = h.constant(1.0)
  for i = n - 1; i >= 0; i = i - 1 {
    if adj[i] == zero {
      continue
    }
    push_adjoints(g, h, b, primal_values, adj, param_adj, i)
  }
  let var_adj : Array[Int] = []
  for slot = 0; slot < g.var_count; slot = slot + 1 {
    // A variable the caller declared but never used has no node and
    // therefore no adjoint: its gradient is exactly zero.
    match g.var_slot(slot) {
      Some(node) => var_adj.push(adj[node])
      None => var_adj.push(zero)
    }
  }
  { graph: h, primal_size: n, var_adj, param_adj_node: param_adj, }
}

///|
/// Build a graph with the same node sequence as `g`, by re-interning
/// every node. Hash-consing makes this nearly free: every node finds
/// its own slot, so the mirrored graph has exactly the same shape.
///
/// Mirroring rather than sharing is deliberate. The backward sweep
/// appends gradient nodes to `h` while leaving `g` untouched, and
/// sharing a single `Array[Node]` between the two would make the
/// primal impossible to keep after differentiation.
pub fn mirror(g : Graph) -> Graph raise ADError {
  let h = Graph::new()
  ignore(h.new_vars(g.var_count))
  for slot = 0; slot < g.param_shapes.length(); slot = slot + 1 {
    ignore(h.new_param(g.param_shapes[slot]))
  }
  for i = 0; i < g.nodes.length(); i = i + 1 {
    ignore(reintern(h, g.nodes[i]))
  }
  // Re-interning the `Var(i)` leaves makes `h`'s own `var_slots`
  // agree with `g`'s, which is what lets a second sweep over the
  // gradient graph find the same variables.
  for slot = 0; slot < g.var_count; slot = slot + 1 {
    ignore(h.variable(slot))
  }
  h
}

///|
/// Re-emit one node into `h`, translating child indices through
/// `child`. The primal's children are already at the same slots in
/// `h`, so the mapping is the identity for the first `primal_size`
/// nodes.
fn reintern(h : Graph, node : Node) -> Int raise ADError {
  match node {
    Node::Const(v) => h.constant(v)
    Node::Var(slot) => h.variable(slot)
    Node::ParamGet(slot, index) => h.param(slot, index)
    Node::Unary(op, a) => h.unary(op, a)
    Node::Binary(op, a, b) => h.binary(op, a, b)
    Node::PowBy(exponent, a) => h.pow(a, exponent)
  }
}

///|
/// Emit the local derivative rules for primal node `i` into `adj`.
///
/// Each rule reads "multiply the incoming adjoint by the local
/// derivative expression, add it to the child's adjoint". The
/// derivative expression is built from the primal's own nodes, so
/// it recomputes correctly when the gradient program is evaluated
/// at a different binding.
fn push_adjoints(
  g : Graph,
  h : Graph,
  b : Binding,
  primal_values : Array[Double],
  adj : Array[Int],
  param_adj : Array[Array[Int]],
  i : Int,
) -> Unit raise ADError {
  let incoming = adj[i]
  match g.nodes[i] {
    Node::Const(_) => ()
    Node::Var(_) => ()
    Node::ParamGet(slot, index) =>
      // A parameter leaf has no primal slot to hang a node off, so
      // its adjoint is accumulated in a side table — but as a node
      // in the gradient graph, not as a number, so that it can be
      // re-evaluated at a different binding later.
      param_adj[slot][index] = h.add(param_adj[slot][index], incoming)
    Node::Unary(op, a) => {
      let d = derivative_expr(h, op, a, primal_values[a])
      adj[a] = h.add(adj[a], h.mul(incoming, d))
    }
    Node::Binary(op, a, b) => {
      let dleft = derivative_left(h, op, a, b)
      let dright = derivative_right(h, op, a, b)
      adj[a] = h.add(adj[a], h.mul(incoming, dleft))
      adj[b] = h.add(adj[b], h.mul(incoming, dright))
    }
    Node::PowBy(exponent, a) => {
      // d/dx x^c = c * x^(c-1)
      let inner = h.pow(a, exponent - 1.0)
      let d = h.mul(h.constant(exponent), inner)
      adj[a] = h.add(adj[a], h.mul(incoming, d))
    }
  }
}

///|
/// The derivative of `op(x)` with respect to `x`, as an expression
/// over the primal graph.
///
/// Every smooth branch reuses the primal's node rather than its
/// value, which is what keeps the gradient re-evaluable at a
/// different binding.
///
/// ReLU is the exception and cannot be one. Its derivative is
/// discontinuous — zero below the kink, one above — and the IR has
/// no comparison to express "if the runtime value is positive". So
/// ReLU reads the primal value and emits a literal 0 or 1. That is a
/// genuine limitation rather than a shortcut, and it is the price
/// of keeping the IR free of control flow: a ReLU's gradient is
/// correct at the point it was compiled and has to be recompiled
/// for another point. The tests pin the behaviour at both sides of
/// the kink so the boundary is documented rather than accidental.
pub fn derivative_expr(
  h : Graph,
  op : UnaryOp,
  x : Int,
  x_value : Double,
) -> Int raise ADError {
  match op {
    UnaryOp::Neg => h.constant(-1.0)
    UnaryOp::Exp => h.exp(x)
    UnaryOp::Ln => h.div(h.constant(1.0), x)
    UnaryOp::Sqrt => h.mul(h.constant(0.5), h.div(h.constant(1.0), h.sqrt(x)))
    UnaryOp::Sin => h.cos(x)
    UnaryOp::Cos => h.neg(h.sin(x))
    UnaryOp::Tanh => {
      // 1 - t^2 with t = tanh(x)
      let t = h.tanh(x)
      let square = h.mul(t, t)
      h.sub(h.constant(1.0), square)
    }
    UnaryOp::Sigmoid => {
      // s(1 - s) with s = sigmoid(x)
      let s = h.sigmoid(x)
      h.mul(s, h.sub(h.constant(1.0), s))
    }
    UnaryOp::Softplus => h.sigmoid(x)
    UnaryOp::Relu =>
      if x_value > 0.0 {
        h.constant(1.0)
      } else {
        h.constant(0.0)
      }
  }
}

///|
/// Partial derivative of `a op b` with respect to `a`.
pub fn derivative_left(
  h : Graph,
  op : BinaryOp,
  a : Int,
  b : Int,
) -> Int raise ADError {
  ignore(a)
  match op {
    BinaryOp::Add => h.constant(1.0)
    BinaryOp::Sub => h.constant(1.0)
    BinaryOp::Mul => b
    BinaryOp::Div => h.div(h.constant(1.0), b)
  }
}

///|
/// Partial derivative of `a op b` with respect to `b`.
pub fn derivative_right(
  h : Graph,
  op : BinaryOp,
  a : Int,
  b : Int,
) -> Int raise ADError {
  match op {
    BinaryOp::Add => h.constant(1.0)
    BinaryOp::Sub => h.constant(-1.0)
    BinaryOp::Mul => a
    BinaryOp::Div => {
      // -a / b^2
      let numerator = h.neg(a)
      let denominator = h.mul(b, b)
      h.div(numerator, denominator)
    }
  }
}

///|
/// Evaluate one node of the gradient program at `b`.
///
/// The trick that keeps this cheap: the gradient graph's first
/// `primal_size` slots are a mirror of the primal, so their values
/// are already sitting in `primal_values` and are read straight out.
/// Only the emitted gradient arithmetic above that line is actually
/// evaluated, and an adjoint reaching a parameter leaf is typically
/// a short chain of multiplies against literals.
pub fn eval_adjoint(
  h : Graph,
  b : Binding,
  primal_values : Array[Double],
  id : Int,
) -> Double {
  if id < 0 || id >= h.len() {
    return 0.0
  }
  if id < primal_values.length() {
    return primal_values[id]
  }
  match h.nodes[id] {
    Node::Const(v) => v
    Node::Var(slot) =>
      if slot >= 0 && slot < b.vars.length() {
        b.vars[slot]
      } else {
        0.0
      }
    Node::ParamGet(slot, index) =>
      if slot < 0 || slot >= b.params.length() {
        0.0
      } else {
        let vec = b.params[slot]
        if index < 0 || index >= vec.length() {
          0.0
        } else {
          vec[index]
        }
      }
    Node::Unary(op, child) => {
      let x = eval_adjoint(h, b, primal_values, child)
      match apply_unary(op, x) {
        Some(v) => v
        None => 0.0
      }
    }
    Node::Binary(op, a, c) => {
      let x = eval_adjoint(h, b, primal_values, a)
      let y = eval_adjoint(h, b, primal_values, c)
      match apply_binary(op, x, y) {
        Some(v) => v
        None => 0.0
      }
    }
    Node::PowBy(exponent, base) => {
      let x = eval_adjoint(h, b, primal_values, base)
      match safe_pow(x, exponent) {
        Some(v) => v
        None => 0.0
      }
    }
  }
}

///|
/// Evaluate a slot that the simplifier guarantees is a literal.
///
/// The backward sweep only ever multiplies adjoints by local
/// derivatives, and every one of those is wrapped in `Const` before
/// being interned. The simplifier may then fold the product into a
/// single literal, so this is the only shape an adjoint can have
/// when it is still "just a number". Anything else would be a bug,
/// and we surface it instead of silently reading a wrong slot.
pub fn Graph::eval_constant(self : Graph, id : Int) -> Double {
  match self.nodes[id] {
    Node::Const(v) => v
    _ => 0.0
  }
}

///|
/// Differentiate `target` with respect to every variable and return
/// the numeric gradients together with the program that produced
/// them.
///
/// ```moonbit nocheck
/// let g = Graph::new()
/// ignore(g.new_vars(2))
/// let x = g.variable(0)
/// let y = g.variable(1)
/// let f = g.mul(g.add(g.pow(x, 2.0), g.constant(3.0)), y)
/// let b = Binding::new()
/// b.set_var(0, 2.0)
/// b.set_var(1, 5.0)
/// let (grads, _prog) = grad(g, b, f)
/// // grads = [30.0, 7.0]  — d/dx = 2xy = 20, d/dy = x^2+3 = 7
/// ```
pub fn grad(
  g : Graph,
  b : Binding,
  target : Int,
) -> (Array[Double], Program) raise ADError {
  let primal_values = forward_values(g, b)
  let compiled = compile_gradient(g, b, primal_values, target)
  let prog = Program::{
    primal: g,
    grad: compiled,
    primal_values,
    output: primal_values[target],
  }
  (evaluate_gradients(prog, b), prog)
}

///|
/// Evaluate a compiled gradient at a (possibly different) binding.
///
/// Because the gradient is a program, a caller can re-evaluate it as
/// often as it likes without re-running the backward sweep. A
/// training loop builds the graph once and calls this every step.
pub fn evaluate_gradients(
  p : Program,
  b : Binding,
) -> Array[Double] raise ADError {
  let out : Array[Double] = []
  for i = 0; i < p.grad.var_adj.length(); i = i + 1 {
    out.push(forward(p.grad.graph, b, p.grad.var_adj[i]).output)
  }
  out
}

///|
/// Parameter gradients of a compiled program, evaluated at `b`.
///
/// Each entry is a partial evaluation of the accumulated adjoint
/// circuit, so the values are correct at any binding rather than at
/// the one the sweep ran against.
pub fn param_gradients(
  p : Program,
  b : Binding,
) -> Array[Array[Double]] raise ADError {
  let out : Array[Array[Double]] = []
  for i = 0; i < p.grad.param_adj_node.length(); i = i + 1 {
    let row : Array[Double] = []
    let src = p.grad.param_adj_node[i]
    for j = 0; j < src.length(); j = j + 1 {
      row.push(eval_adjoint(p.grad.graph, b, p.primal_values, src[j]))
    }
    out.push(row)
  }
  out
}

///|
/// Parameter gradients as one flat array, in slot-then-index order,
/// which is the layout an optimiser wants.
pub fn flat_param_gradients(
  p : Program,
  b : Binding,
) -> Array[Double] raise ADError {
  let nested = param_gradients(p, b)
  let out : Array[Double] = []
  for i = 0; i < nested.length(); i = i + 1 {
    for j = 0; j < nested[i].length(); j = j + 1 {
      out.push(nested[i][j])
    }
  }
  out
}
