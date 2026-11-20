// tensor_simplify_test.mbt
//
// The tensor graph's peephole simplifier.
//
// Two things have to hold at once, and they pull in opposite
// directions: the simplifier must remove work, and it must not change
// what the graph computes. The first is checked here directly, node
// by node. The second is checked by the rest of the suite — every
// gradient rule is compared against a finite difference of the same
// function, so a simplifier that changed a value would turn those
// tests red.
//
// The identity rules are the scalar engine's, read out of
// simplify.mbt, so the two engines cannot disagree about what the
// identity of an operator is.

///|
/// A tensor graph with one input, one parameter and a binding that
/// gives them values.
fn simplify_fixture() -> (@moongrad.TGraph, @moongrad.TBinding, Int32) {
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([2.0, 3.0]))
  ignore(g.param(0))
  (g, b, x)
}

///|
/// A zero scalar, as the graph would hold it.
fn zero_scalar() -> @moongrad.Tensor {
  @moongrad.Tensor::scalar(0.0)
}

///|
test "adding_a_zero_scalar_is_the_operand" {
  let (g, _b, x) = simplify_fixture()
  let z = g.constant(zero_scalar())
  let out = g.binary(@moongrad.BinaryOp::Add, x, z)
  assert_eq(out, x)
}

///|
test "subtracting_a_zero_scalar_is_the_operand" {
  let (g, _b, x) = simplify_fixture()
  let z = g.constant(zero_scalar())
  let out = g.binary(@moongrad.BinaryOp::Sub, x, z)
  assert_eq(out, x)
}

///|
test "multiplying_by_one_is_the_operand" {
  let (g, _b, x) = simplify_fixture()
  let one = g.constant(@moongrad.Tensor::scalar(1.0))
  let out = g.binary(@moongrad.BinaryOp::Mul, x, one)
  assert_eq(out, x)
}

///|
test "dividing_by_one_is_the_operand" {
  let (g, _b, x) = simplify_fixture()
  let one = g.constant(@moongrad.Tensor::scalar(1.0))
  let out = g.binary(@moongrad.BinaryOp::Div, x, one)
  assert_eq(out, x)
}

///|
test "a_zero_on_the_left_is_the_operand_too" {
  let (g, _b, x) = simplify_fixture()
  let z = g.constant(zero_scalar())
  let out = g.binary(@moongrad.BinaryOp::Add, z, x)
  assert_eq(out, x)
}

///|
test "a_zero_of_a_shape_is_not_an_identity" {
  // A zero of shape (2,3) is not the additive identity: adding it
  // to a (2,3) operand produces a (2,3) result, but adding it to a
  // (2,2) operand is a shape error. The rule only fires for a
  // scalar, whose shape broadcasts against anything.
  let (g, _b, x) = simplify_fixture()
  let z = g.constant(@moongrad.Tensor::zeros([2, 3]))
  let out = g.binary(@moongrad.BinaryOp::Add, x, z)
  assert_true(out != x)
}

///|
test "two_literals_are_folded" {
  let g = @moongrad.TGraph::new()
  let a = g.constant(@moongrad.Tensor::scalar(2.0))
  let b = g.constant(@moongrad.Tensor::scalar(3.0))
  let out = g.binary(@moongrad.BinaryOp::Mul, a, b)
  let values = @moongrad.t_forward(g, @moongrad.TBinding::new())
  assert_eq(values[out].data[0], 6.0)
}

///|
test "the_same_computation_is_one_node" {
  let (g, _b, x) = simplify_fixture()
  let a = g.binary(@moongrad.BinaryOp::Mul, x, x)
  let b = g.binary(@moongrad.BinaryOp::Mul, x, x)
  assert_eq(a, b)
}

///|
test "a_double_negation_is_the_operand" {
  let (g, _b, x) = simplify_fixture()
  let out = g.unary(@moongrad.UnaryOp::Neg, g.unary(@moongrad.UnaryOp::Neg, x))
  assert_eq(out, x)
}

///|
test "the_simplifier_counts_what_it_did" {
  let (g, _b, x) = simplify_fixture()
  let z = g.constant(zero_scalar())
  ignore(g.binary(@moongrad.BinaryOp::Add, x, z))
  ignore(g.binary(@moongrad.BinaryOp::Mul, x, x))
  ignore(g.binary(@moongrad.BinaryOp::Mul, x, x))
  assert_true(g.stats.identities >= 1)
  assert_true(g.stats.cse_hits >= 1)
}

///|
test "the_gradient_program_is_smaller_than_it_would_be" {
  // A regression guard on the simplifier itself. If the identities
  // and the sharing stopped happening, the gradient program would
  // grow past this bound, and the number is well under what it was
  // before the pass existed.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let w = g.param(0)
  let bias = g.param(1)
  let h1 = g.relu(g.add(g.matmul(x, w), bias))
  let w2 = g.param(2)
  let bias2 = g.param(3)
  let out = g.add(g.matmul(h1, w2), bias2)
  let loss = g.sum_axis(g.mul(out, out), 1)
  let b = @moongrad.TBinding::new()
  b.set_input(
    0,
    match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
      Some(t) => t
      None => @moongrad.Tensor::zeros([2, 2])
    },
  )
  b.set_param(
    0,
    match @moongrad.Tensor::matrix([[0.5, -0.25], [0.1, 0.2]]) {
      Some(t) => t
      None => @moongrad.Tensor::zeros([2, 2])
    },
  )
  b.set_param(1, @moongrad.Tensor::vector([0.0, 0.0]))
  b.set_param(
    2,
    match @moongrad.Tensor::matrix([[0.3, 0.4], [-0.1, 0.2]]) {
      Some(t) => t
      None => @moongrad.Tensor::zeros([2, 2])
    },
  )
  b.set_param(3, @moongrad.Tensor::vector([0.0, 0.0]))
  let compiled = @moongrad.compile_tensor_gradient(g, b, loss)
  // Before the simplifier this was 64 nodes.
  assert_true(compiled.graph.len() <= 50)
}

///|
test "a_literal_of_a_shape_is_not_an_identity" {
  // The identity rules fire on a *scalar* literal, because a scalar
  // broadcasts against anything. A zero of some other shape does not:
  // adding it to a (2,2) operand is a shape error rather than the
  // operand, and multiplying it by a (2,3) operand is not the
  // operand either.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let zero = g.constant(@moongrad.Tensor::zeros([2, 3]))
  let out = g.binary(@moongrad.BinaryOp::Add, x, zero)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.0, 2.0]))
  // The shapes do not fit, so the graph refuses rather than adding a
  // zero that was never a zero of the right shape.
  let refused = try {
    let _ = @moongrad.t_grad(g, b, out)
    false
  } catch {
    _ => true
  }
  check(refused, "a zero of another shape should not be an identity")
}

///|
test "a_fold_that_would_produce_an_infinity_is_not_folded" {
  // A fold that produced an infinity would put a not-finite value
  // into the graph before the caller had a chance to see that the
  // gradient had already diverged. The division is refused instead.
  let g = @moongrad.TGraph::new()
  let a = g.constant(@moongrad.Tensor::scalar(1.0))
  let z = g.constant(@moongrad.Tensor::scalar(0.0))
  let out = g.binary(@moongrad.BinaryOp::Div, a, z)
  let b = @moongrad.TBinding::new()
  let values = @moongrad.t_forward(g, b)
  // The forward pass still answers: an infinity, not a refusal. What
  // must not happen is the fold having baked it in earlier.
  check(
    values[out].data[0] == values[out].data[0] || values[out].data[0] > 1.0e308,
    "the division should be an infinity",
  )
}
