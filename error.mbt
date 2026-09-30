// error.mbt
//
// Every failure mode in moongrad is a value, not a trap. A
// differentiation engine is usually wired into a training loop that
// runs unattended for hours, so it cannot be allowed to abort on a
// bad shape or a non-finite gradient; it has to hand the problem
// back to the caller and keep going.
//
// The three error families below mirror the three places a user can
// go wrong: building a malformed graph, asking for a derivative that
// does not exist, and overflowing the numeric tower.

///|
/// Raised by the graph builder and the differentiation passes.
pub suberror ADError {
  /// A node referenced a child slot that has not been interned.
  DanglingNode(Int)
  /// A node requested an operand count that its opcode does not accept.
  BadArity(UnaryOp, Int)
  /// A `Var` slot was referenced that was never declared.
  UnknownVar(Int)
  /// A `ParamGet` referenced a parameter slot that was never declared.
  UnknownParam(Int)
  /// An index into a parameter slot was out of range.
  ParamIndexOutOfRange(Int, Int, Int)
  /// A gradient was requested for a variable that the target does not
  /// actually depend on. Not fatal — the gradient is zero — but it is
  /// almost always a bug in the caller's model, so we surface it.
  DisconnectedVar(Int)
  /// Division by zero, or `log`/`sqrt` of a non-positive number.
  DomainError(String)
  /// A value left the representable range during a pass.
  NonFinite(String)
} derive(Debug)

///|
/// Human-readable form, used by the REPL and the example binaries.
pub fn ADError::describe(self : ADError) -> String {
  match self {
    DanglingNode(id) => "node \{id} references an interned slot that does not exist"
    BadArity(op, got) => "unary operator \{op.symbol()} applied to \{got} operands"
    UnknownVar(i) => "variable slot \{i} was never declared"
    UnknownParam(i) => "parameter slot \{i} was never declared"
    ParamIndexOutOfRange(slot, idx, len) =>
      "parameter slot \{slot} has \{len} entries; index \{idx} is out of range"
    DisconnectedVar(i) =>
      "variable \{i} does not reach the differentiated node; its gradient is 0"
    DomainError(what) => "domain error: \{what}"
    NonFinite(what) => "non-finite value produced by \{what}"
  }
}

///|
/// True when `x` is neither NaN nor an infinity. Gradients that leave
/// the finite reals almost always indicate a diverged training run,
/// and detecting it early is cheaper than debugging the divergence
/// twenty epochs later.
pub fn is_finite(x : Double) -> Bool {
  x == x && x != infinity_pos() && x != neg_infinity_pos()
}

///|
fn infinity_pos() -> Double {
  1.0 / 0.0
}

///|
fn neg_infinity_pos() -> Double {
  -1.0 / 0.0
}
