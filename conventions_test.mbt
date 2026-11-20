// conventions_test.mbt
//
// Conventions, scales and folded constants.
//
// The other test files check the algebra: that a rule is right, that a
// gradient matches a finite difference, that a loss falls. They do not
// check the things a rule can be wrong about without being wrong
// algebraically — the sign convention of a Cholesky factor, the
// absolute size of a reported loss, the value a constant folds to.
//
// A mutation audit found exactly these gaps: changing the folded value
// of `exp(0)` from one to two, halving the trainer's objective, or
// negating the diagonal of a Cholesky factor all left the suite green.
// The tests below close those holes, and each of them fails against the
// code as it was before the corresponding fix.

///|
/// A graph whose only node is a constant.
fn constant_graph(v : Double) -> (@moongrad.Graph, @moongrad.Binding, Int32) {
  let g = @moongrad.Graph::new()
  let b = @moongrad.Binding::new()
  let x = g.constant(v)
  (g, b, x)
}

///|
test "the_folded_values_of_the_operators_are_the_right_ones" {
  // The constant folder has special cases for a few arguments, and
  // one of them was wrong: the hyperbolic tangent of minus one was
  // folded to minus one, rather than to minus 0.7615941559557649. The
  // tensor engine folded it correctly, so the two engines disagreed
  // about the value of a constant.
  let cases : Array[(@moongrad.UnaryOp, Double, Double)] = [
    (@moongrad.UnaryOp::Exp, 0.0, 1.0),
    (@moongrad.UnaryOp::Ln, 1.0, 0.0),
    (@moongrad.UnaryOp::Tanh, -1.0, -0.7615941559557649),
    (@moongrad.UnaryOp::Tanh, 0.0, 0.0),
    (@moongrad.UnaryOp::Sigmoid, 0.0, 0.5),
    (@moongrad.UnaryOp::Relu, 0.0, 0.0),
  ]
  for i = 0; i < cases.length(); i = i + 1 {
    let (op, argument, expected) = cases[i]
    let (g, b, node) = constant_graph(argument)
    let folded = g.unary(op, node)
    let values = @moongrad.forward_values(g, b)
    assert_close(values[folded], expected, 1.0e-12, "the folded value")
  }
}

///|
test "the_relu_derivative_at_the_kink_is_zero_in_the_scalar_engine" {
  // The convention is written down: the derivative of the rectified
  // linear unit is defined as zero at exactly zero. Three places
  // implement it — the reverse engine, the forward-mode oracle and
  // the dual numbers — and each of them could have been changed on
  // its own without the suite noticing.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let y = g.relu(x)
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.0)
  let (grads, _program) = @moongrad.grad(g, b, y)
  assert_close(grads[0], 0.0, 1.0e-12, "the derivative at the kink")
}

///|
test "the_cholesky_factor_has_a_positive_diagonal" {
  // A Cholesky factor is the lower triangle with a positive
  // diagonal, and the reconstruction `L L^T = A` holds either way.
  // A factor with a negative diagonal is not a Cholesky factor, and
  // a caller who took the square root of the diagonal would get the
  // wrong sign.
  let a = match @moongrad.Tensor::matrix([[4.0, 2.0], [2.0, 3.0]]) {
    Some(t) => t
    None => return
  }
  let l = match @moongrad.cholesky(a) {
    Some(t) => t
    None => {
      check(false, "the matrix is positive definite")
      return
    }
  }
  for i = 0; i < 2; i = i + 1 {
    check(l.mget(i, i) > 0.0, "the diagonal should be positive")
  }
}

///|
test "the_solver_pivots_when_the_first_pivot_is_zero" {
  // The matrix [[0,1],[1,0]] needs a row swap before it can be
  // eliminated: its first pivot is zero. A solver that did not look
  // for a better pivot would divide by zero and hand back a NaN
  // where the answer is perfectly well conditioned.
  let a = match @moongrad.Tensor::matrix([[0.0, 1.0], [1.0, 0.0]]) {
    Some(t) => t
    None => return
  }
  let x = match @moongrad.solve(a, [1.0, 2.0]) {
    Some(v) => v
    None => {
      check(false, "the solve should have worked")
      return
    }
  }
  assert_close(x[0], 2.0, 1.0e-9, "the first unknown")
  assert_close(x[1], 1.0, 1.0e-9, "the second unknown")
}

///|
test "the_trainer_reports_the_mean_squared_error_of_the_model" {
  // The trainer's other tests checked that the loss falls, which a
  // scale error leaves untouched: halving the objective halves the
  // first loss and the last one, and the comparison still holds. The
  // absolute number has to be right as well.
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 4, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    0,
  )
  // The forward pass at the weights the trainer starts from, worked
  // out here rather than taken from the trainer.
  let b = @moongrad.trainer_binding(
    trainer.graph,
    trainer.data,
    trainer.flat,
    trainer.net,
  )
  let values = @moongrad.forward_values(trainer.graph, b)
  let expected = values[trainer.out]
  let report = trainer.train_step(-1.0)
  assert_close(report.loss, expected, 1.0e-12, "the reported loss")
}

///|
test "a_learning_rate_override_takes_effect" {
  // The override replaces the optimiser for that step. It was only
  // ever exercised with zero, where "the override does nothing" and
  // "the override is ignored" look exactly the same.
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 4, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    0,
  )
  let before = trainer.flat.copy()
  ignore(trainer.train_step(0.0))
  let mut moved = false
  for i = 0; i < before.length(); i = i + 1 {
    if trainer.flat[i] != before[i] {
      moved = true
    }
  }
  check(!moved, "a rate of zero should leave the weights alone")
  ignore(trainer.train_step(0.5))
  let mut moved = false
  for i = 0; i < before.length(); i = i + 1 {
    if trainer.flat[i] != before[i] {
      moved = true
    }
  }
  check(moved, "a positive override should move the weights")
}

///|
test "the_shared_away_count_matches_the_slots_saved" {
  // Two counters say how much the simplifier saved. One of them was
  // never checked against the other.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let y = g.add(g.mul(x, x), g.constant(0.0))
  ignore(y)
  check(
    g.stats().shared_away() == g.stats().slots_saved(),
    "the two counters should agree",
  )
}
