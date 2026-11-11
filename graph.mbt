// graph.mbt
//
// The builder. Every arithmetic operation in a moongrad program goes
// through a `Graph` method, which does three things in order:
//
//   1. simplify — recognise identities (`x + 0`, `x * 1`, `x - x`, …)
//      and constant-fold subtrees whose children are both literals;
//   2. intern — look the resulting node up in a hash table and reuse
//      an existing slot if an identical node is already there;
//   3. append — otherwise give it a fresh slot at the end.
//
// Step 1 shrinks the graph before it grows, step 2 makes the graph a
// DAG rather than a tree. Both matter for a training loop that
// rebuilds the same expression thousands of times per epoch: the
// difference is visible in `Stats` and reported by
// `examples/cse_savings`.
//
// Because the simplifier runs before interning, two structurally
// identical computations that differ only in algebraic form
// (`a + b + b + a` versus `2 * (a + b)`) still land on the same slot.

///|
/// Counters describing what the builder did. These are not
/// diagnostics; they are the measurable output of the optimisation
/// passes, and the examples assert on them.
#derive(Debug, Eq)
pub(all) struct Stats {
  /// Slots handed out. Equals the length of `Graph::nodes`.
  mut nodes_built : Int32
  /// Times interning returned an existing slot. Each hit is a
  /// subexpression that would otherwise have been computed twice.
  mut cse_hits : Int32
  /// Times a node was replaced by a peephole identity (`x*1` → `x`).
  mut identities : Int32
  /// Times a subtree was evaluated at build time because all of its
  /// leaves were literals.
  mut folds : Int32
  /// Times a node was shortened by an annihilator (`x*0` → `0`).
  mut annihilated : Int32
  /// Builder calls that were made. A call that hits the intern
  /// table still counts here, so the difference between this and
  /// `nodes_built` is exactly what sharing saved.
  mut requests : Int32
}

///|
pub extend Stats with Debug::{to_repr}

///|
pub extend Stats with Eq::{equal, not_equal}

///|
/// An empty `Stats`.
pub fn Stats::new() -> Stats {
  {
    nodes_built: 0,
    cse_hits: 0,
    identities: 0,
    folds: 0,
    annihilated: 0,
    requests: 0,
  }
}

///|
/// Structural complexity of the graph as it stands: one unit per
/// distinct node. A naive builder that skipped both simplification
/// and interning would report a much larger number for the same
/// program, which is exactly the comparison
/// `examples/cse_savings` makes.
pub fn Stats::distinct_nodes(self : Stats) -> Int32 {
  self.nodes_built
}

///|
/// Number of builder calls that found an existing slot. This is
/// the size of the sharing win: a naive builder that allocated a
/// fresh node for every operation would have needed this many more
/// slots than the graph actually has.
pub fn Stats::shared_away(self : Stats) -> Int32 {
  self.cse_hits + self.identities + self.folds + self.annihilated
}

///|
/// Total subexpressions that sharing removed. `cse_hits` plus
/// `identities` plus `annihilated` plus `folds` is the number of
/// node slots the naive builder would have needed.
pub fn Stats::slots_saved(self : Stats) -> Int32 {
  self.cse_hits + self.identities + self.folds + self.annihilated
}

///|
/// A computation under construction.
///
/// `Graph` is mutable on purpose. It is a builder: a program is
/// assembled once, then differentiated many times, and threading a
/// fresh `Graph` value through every intermediate expression would
/// be noise without adding safety. Everything the user can observe —
/// `nodes`, `var_count`, `param_shapes` — is read-only.
#derive(Debug)
pub(all) struct Graph {
  mut nodes : Array[Node]
  /// Slot of the `Var(i)` node for each declared variable. Tracked
  /// separately because `Var(i)` is interned lazily: a variable that
  /// the caller never referenced in an expression has no node at all
  /// until something asks for one, and asking must not silently
  /// append above the region the reverse sweep is walking.
  var_slots : Array[Int32]
  mut var_count : Int32
  mut param_shapes : Array[Int32]
  /// Hash-cons table. Maps a node shape to the slot that already
  /// holds it.
  intern_table : @hashmap.HashMap[Node, Int32]
  mut stats : Stats
}

///|
pub extend Graph with Debug::{to_repr}

///|
/// A graph with no nodes and no inputs.
pub fn Graph::new() -> Graph {
  {
    nodes: [],
    var_slots: [],
    var_count: 0,
    param_shapes: [],
    intern_table: @hashmap.HashMap([]),
    stats: Stats::new(),
  }
}

///|
/// Number of slots in the graph. Since every edge points to a lower
/// index, this is also the index of the next slot to be handed out.
pub fn Graph::len(self : Graph) -> Int32 {
  self.nodes.length()
}

///|
/// Declare `n` differentiation variables and return the index of the
/// first one. Slots are numbered consecutively, so a graph with three
/// variables exposes `0, 1, 2`.
pub fn Graph::new_vars(self : Graph, n : Int32) -> Int32 {
  let first = self.var_count
  self.var_count = self.var_count + n
  first
}

///|
/// The interned slot of variable `i`, or `None` when the caller
/// never mentioned it. A gradient of `None` is zero by definition.
pub fn Graph::var_slot(self : Graph, i : Int32) -> Int32? {
  if i < 0 || i >= self.var_slots.length() {
    None
  } else if self.var_slots[i] < 0 {
    // The variable was declared but never used, so it has no node
    // and its gradient is zero.
    None
  } else {
    Some(self.var_slots[i])
  }
}

///|
/// Declare a parameter vector of `size` scalars and return its slot.
pub fn Graph::new_param(self : Graph, size : Int32) -> Int32 {
  let slot = self.param_shapes.length()
  self.param_shapes.push(size)
  slot
}

///|
/// Read a `Graph` slot. Returns the node so callers can inspect the
/// program; differentiation passes use the index form below.
pub fn Graph::get(self : Graph, id : Int32) -> Node raise ADError {
  if id < 0 || id >= self.nodes.length() {
    raise DanglingNode(id)
  }
  self.nodes[id]
}

///|
/// Append `node` unless an identical node is already present, in
/// which case return the existing slot. This is the hash-cons
/// operation; every other builder method funnels through it.
pub fn Graph::intern(self : Graph, node : Node) -> Int32 {
  self.stats.requests = self.stats.requests + 1
  match self.intern_table.get(node) {
    Some(existing) => {
      self.stats.cse_hits = self.stats.cse_hits + 1
      existing
    }
    None => {
      let id = self.nodes.length()
      self.nodes.push(node)
      self.intern_table.set(node, id)
      self.stats.nodes_built = self.stats.nodes_built + 1
      id
    }
  }
}

///|
/// Emit a literal. Literals are interned too, so the graph of a
/// program that mentions `2.0` a hundred times stores it once.
pub fn Graph::constant(self : Graph, value : Double) -> Int32 {
  self.intern(Node::Const(value))
}

///|
/// The slot of differentiation variable `i`, interning it on first
/// use. The slot is remembered in `var_slots` so the reverse sweep
/// can find it again without having to intern during the sweep —
/// interning mid-sweep would append above the region being walked.
pub fn Graph::variable(self : Graph, i : Int32) -> Int32 raise ADError {
  if i < 0 || i >= self.var_count {
    raise UnknownVar(i)
  }
  if i < self.var_slots.length() {
    if self.var_slots[i] >= 0 {
      return self.var_slots[i]
    }
  }
  let slot = self.intern(Node::Var(i))
  // The slots are indexed by the variable they belong to, not by the
  // order the variables were first used in. A graph that uses its
  // variables out of order would otherwise hand one variable's
  // gradient to another.
  while self.var_slots.length() < i {
    self.var_slots.push(-1)
  }
  self.var_slots.push(slot)
  slot
}

///|
/// Read element `index` of parameter vector `slot`.
pub fn Graph::param(
  self : Graph,
  slot : Int32,
  index : Int32,
) -> Int32 raise ADError {
  if slot < 0 || slot >= self.param_shapes.length() {
    raise UnknownParam(slot)
  }
  let size = self.param_shapes[slot]
  if index < 0 || index >= size {
    raise ParamIndexOutOfRange(slot, index, size)
  }
  self.intern(Node::ParamGet(slot, index))
}

///|
/// Build `op(child)` after simplification.
pub fn Graph::unary(self : Graph, op : UnaryOp, child : Int32) -> Int32 {
  self.simplify_unary(op, child)
}

///|
/// Build `left op right` after simplification and constant folding.
pub fn Graph::binary(
  self : Graph,
  op : BinaryOp,
  left : Int32,
  right : Int32,
) -> Int32 {
  self.simplify_binary(op, left, right)
}

///|
/// Shorthand for `Add`.
pub fn Graph::add(self : Graph, a : Int32, b : Int32) -> Int32 {
  self.binary(BinaryOp::Add, a, b)
}

///|
/// Shorthand for `Sub`.
pub fn Graph::sub(self : Graph, a : Int32, b : Int32) -> Int32 {
  self.binary(BinaryOp::Sub, a, b)
}

///|
/// Shorthand for `Mul`.
pub fn Graph::mul(self : Graph, a : Int32, b : Int32) -> Int32 {
  self.binary(BinaryOp::Mul, a, b)
}

///|
/// Shorthand for `Div`.
pub fn Graph::div(self : Graph, a : Int32, b : Int32) -> Int32 {
  self.binary(BinaryOp::Div, a, b)
}

///|
/// `a ^ c` for a real exponent. The exponent rides along inside the
/// node so that the reverse pass can apply `c * a^(c-1)` without a
/// side table.
pub fn Graph::pow(self : Graph, a : Int32, c : Double) -> Int32 {
  self.simplify_pow(c, a)
}

// Shorthands for the unary operators. Each is `unary(op, child)`;
// spelling them out at every call site is noise, and the
// mathematical names read better in the formulas that follow.

///|
pub fn Graph::neg(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Neg, x)
}

///|
pub fn Graph::exp(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Exp, x)
}

///|
pub fn Graph::ln(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Ln, x)
}

///|
pub fn Graph::sqrt(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Sqrt, x)
}

///|
pub fn Graph::sin(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Sin, x)
}

///|
pub fn Graph::cos(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Cos, x)
}

///|
pub fn Graph::tanh(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Tanh, x)
}

///|
pub fn Graph::sigmoid(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Sigmoid, x)
}

///|
pub fn Graph::softplus(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Softplus, x)
}

///|
/// ReLU. `max(0, x)`, with the derivative defined as 0 at exactly
/// zero — the standard convention, and the one that keeps a unit
/// that lands exactly on the boundary from producing a gradient of
/// 0.5 and stalling.
pub fn Graph::relu(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Relu, x)
}

///|
/// The sigmoid linear unit, `x * sigmoid(x)`. Also called swish.
pub fn Graph::silu(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Silu, x)
}

///|
/// The Gaussian error linear unit, in its tanh approximation.
pub fn Graph::gelu(self : Graph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Gelu, x)
}

///|
/// Build the derivative of ReLU as a node.
///
/// The node is not simplified: it has no constant form, because the
/// point where the derivative is one and the point where it is zero
/// are decided when the graph is evaluated.
pub fn Graph::relu_deriv(self : Graph, x : Int32) -> Int32 {
  self.intern(Node::ReluDeriv(x))
}

///|
/// The literal stored in slot `id`, or `None`. Used by
/// `derivative_expr` for operators whose derivative is not an
/// ordinary smooth function of the operand — ReLU is the only one,
/// and the discontinuity means the branch has to be decided at
/// evaluation time rather than folded into a literal.
pub fn Graph::eval_literal(self : Graph, id : Int32) -> Double {
  match self.literal_of(id) {
    Some(v) => v
    None =>
      // Not a literal. Reading the forward value would require a
      // binding we do not have here, so fall back to the
      // non-positive branch, which is the one that keeps a
      // saturated unit from passing gradient through.
      0.0
  }
}
