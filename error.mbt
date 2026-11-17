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
#derive(Debug)
pub suberror ADError {
  /// A node referenced a child slot that has not been interned.
  DanglingNode(Int32)
  /// A `Var` slot was referenced that was never declared.
  UnknownVar(Int32)
  /// A `ParamGet` referenced a parameter slot that was never declared.
  UnknownParam(Int32)
  /// An index into a parameter slot was out of range.
  ParamIndexOutOfRange(Int32, Int32, Int32)
  /// Division by zero, or `log`/`sqrt` of a non-positive number.
  DomainError(String)
  /// A compiled gradient was evaluated against inputs whose shapes
  /// are not the ones it was compiled for.
  ///
  /// The gradient of a broadcast sum is summed back down over the
  /// axes that were broadcast, so a program evaluated against
  /// differently-shaped inputs would quietly compute the gradient of
  /// a different function.
  ShapeMismatch(String, String)
}

///|
pub extend ADError with Debug::{to_repr}

///|
/// Human-readable form, used by the REPL and the example binaries.
pub fn ADError::describe(self : ADError) -> String {
  match self {
    DanglingNode(id) =>
      "node \{id} references an interned slot that does not exist"
    UnknownVar(i) => "variable slot \{i} was never declared"
    UnknownParam(i) => "parameter slot \{i} was never declared"
    ParamIndexOutOfRange(slot, idx, len) =>
      "parameter slot \{slot} has \{len} entries; index \{idx} is out of range"
    DomainError(what) => "domain error: \{what}"
    ShapeMismatch(expected, got) =>
      "the gradient was compiled for inputs of shape \{expected} and was evaluated against \{got}"
  }
}

///|
/// True when `x` is neither NaN nor an infinity. Gradients that leave
/// the finite reals almost always indicate a diverged training run,
/// and detecting it early is cheaper than debugging the divergence
/// twenty epochs later.
///
/// This is the standard library's test; the name is kept because the
/// library's own error convention talks about finite and non-finite
/// values in these terms.
pub fn is_finite(x : Double) -> Bool {
  x.is_finite()
}
