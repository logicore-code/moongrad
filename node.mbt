// node.mbt
//
// The intermediate representation. A moongrad computation is a DAG
// whose nodes live in a flat `Array[Node]`; a node is referred to
// everywhere else by its index in that array.
//
// The flat-array representation is not an implementation detail, it
// is the reason the whole engine is simple. Because a node can only
// be built after its children exist, every edge of the graph points
// from a higher index to a lower one. The forward pass is therefore
// a left-to-right scan and the reverse pass is a right-to-left scan.
// Neither needs a topological sort, and neither can get the order
// wrong.
//
// Every variant derives `Hash` and `Eq` so that the graph can be
// hash-consed: interning an identical node twice returns the first
// slot, which is what buys common-subexpression elimination for free
// (see `graph.mbt`).

///|
/// Unary elementwise operations. `Sigmoid` and `Softplus` are here
/// rather than in the neural-network layer because they are closed
/// under differentiation with a cheap local derivative, and because
/// cross-entropy in the numerical-stable form is
/// `softplus(z) - z*y`, so it is nice to have both available when
/// the caller assembles a loss by hand.
pub(all) enum UnaryOp {
  Neg
  Exp
  Ln
  Sqrt
  Sin
  Cos
  Tanh
  Sigmoid
  Softplus
} derive(Hash, Eq, Debug)

///|
/// Infix operators with constant arity.
pub(all) enum BinaryOp {
  Add
  Sub
  Mul
  Div
} derive(Hash, Eq, Debug)

///|
/// One slot of the computation graph.
pub(all) enum Node {
  /// A literal. The simplifier folds these aggressively.
  Const(Double)
  /// An input to be differentiated. Slots are declared up front via
  /// `Graph::new_var` and are the only source of non-zero gradients.
  Var(Int)
  /// A read of a model parameter. Parameters are differentiated too;
  /// they just arrive pre-baked rather than as function arguments.
  ParamGet(Int, Int)
  Unary(UnaryOp, Int)
  Binary(BinaryOp, Int, Int)
  /// `base ^ exponent`, with the exponent carried inline as a
  /// literal. Keeping it in the node means the derivative rule
  /// `d/dx x^c = c * x^(c-1)` needs no table lookup, and it keeps
  /// `PowBy` at arity one so the graph stays a tree in that
  /// direction.
  PowBy(Double, Int)
} derive(Hash, Eq, Debug)

///|
/// Symbol used when printing a graph in infix form.
pub fn UnaryOp::symbol(self : UnaryOp) -> String {
  match self {
    Neg => "-"
    Exp => "exp"
    Ln => "log"
    Sqrt => "sqrt"
    Sin => "sin"
    Cos => "cos"
    Tanh => "tanh"
    Sigmoid => "sigmoid"
    Softplus => "softplus"
  }
}

///|
/// Symbol used when printing a graph in infix form.
pub fn BinaryOp::symbol(self : BinaryOp) -> String {
  match self {
    Add => "+"
    Sub => "-"
    Mul => "*"
    Div => "/"
  }
}

///|
/// The identity element for a binary operator, used by the
/// simplifier when folding a node against a constant.
pub fn BinaryOp::identity(self : BinaryOp) -> Double {
  match self {
    Add => 0.0
    Sub => 0.0
    Mul => 1.0
    Div => 1.0
  }
}

///|
/// True when multiplying by the annihilator of `self` kills the other
/// operand outright. Only `Mul` qualifies, and only because `0 * x`
/// is zero for every finite `x`; the other three would need a
/// division, so they are left to the constant folder.
pub fn BinaryOp::kills_product(self : BinaryOp) -> Bool {
  match self {
    Mul => true
    _ => false
  }
}

///|
/// Number of children a node of this shape refers to.
pub fn Node::arity(self : Node) -> Int {
  match self {
    Const(_) => 0
    Var(_) => 0
    ParamGet(_, _) => 0
    Unary(_, _) => 1
    Binary(_, _, _) => 2
    PowBy(_, _) => 1
  }
}

///|
/// Children of a node, in evaluation order. The reverse pass walks
/// these to push adjoints, so the order matters only for
/// deterministic accumulation; the result is the same either way
/// because addition of adjoints is commutative.
pub fn Node::children(self : Node) -> Array[Int] {
  match self {
    Const(_) => []
    Var(_) => []
    ParamGet(_, _) => []
    Unary(_, a) => [a]
    Binary(_, a, b) => [a, b]
    PowBy(_, a) => [a]
  }
}

///|
/// True when the node's value is known without evaluating any
/// child. Only `Const` qualifies; leaves depend on the caller's
/// binding, so the simplifier never folds them.
pub fn Node::is_literal(self : Node) -> Bool {
  match self {
    Const(_) => true
    _ => false
  }
}
