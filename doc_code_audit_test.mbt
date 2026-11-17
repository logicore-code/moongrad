// doc_code_audit_test.mbt
//
// Regression tests for the places where the documentation and the
// code disagreed. Each test here fails against the code as it was
// before the fix, and each one pins a behaviour that the
// documentation now claims.
//
// The pattern is the same throughout: a function that had two
// answers — the documented one and a silent fallback — now has one,
// and the test is the one that says which.

///|
/// A binding with the given variable values, one per slot.
fn audit_binding_of(vars : Array[Double]) -> @moongrad.Binding {
  let b = @moongrad.Binding::new()
  for i = 0; i < vars.length(); i = i + 1 {
    b.set_var(i, vars[i])
  }
  b
}

///|
/// An empty dataset. It is well formed: there is nothing to be
/// ragged about.
fn audit_empty_dataset() -> @moongrad.Dataset {
  match @moongrad.Dataset::new([], [], 1) {
    Some(d) => d
    None => @moongrad.Dataset::new([], [], 1).unwrap()
  }
}

///|
/// A dataset of one sample with two features.
fn audit_one_sample_dataset() -> @moongrad.Dataset {
  match @moongrad.Dataset::new([[1.0, 2.0]], [1.0], 1) {
    Some(d) => d
    None => @moongrad.Dataset::new([[1.0, 2.0]], [1.0], 1).unwrap()
  }
}

// ---------------------------------------------------------------------------
// Reading a slot
// ---------------------------------------------------------------------------

///|
test "eval_constant_reads_a_literal_slot" {
  let g = @moongrad.Graph::new()
  let c = g.constant(2.5)
  assert_eq(g.eval_constant(c), 2.5)
}

///|
test "eval_constant_refuses_a_slot_that_is_not_a_literal" {
  // The documentation said that anything else was surfaced as a
  // bug. The code returned zero, which is a number the caller
  // cannot tell from a real answer.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let raised = try {
    let _ = g.eval_constant(x)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "eval_literal_reports_a_slot_that_is_not_a_literal" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  assert_eq(g.eval_literal(g.constant(2.5)), Some(2.5))
  assert_eq(g.eval_literal(x), None)
  // A slot that does not exist is not a literal either.
  assert_eq(g.eval_literal(999), None)
}

// ---------------------------------------------------------------------------
// The forward-mode oracle
// ---------------------------------------------------------------------------

///|
test "the_forward_mode_refuses_a_binding_without_the_variable" {
  // The oracle used to answer "the gradient is zero" for a
  // malformed binding, which is the one class of disagreement it
  // exists to catch.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.mul(x, x)
  let raised = try {
    let _ = @moongrad.forward_mode(g, @moongrad.Binding::new(), f, 0)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "the_dual_engine_refuses_an_unknown_variable_slot" {
  let b = @moongrad.Binding::new()
  let raised = try {
    let _ = @moongrad.Dual::from_var(b, 0, 0)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "the_dual_engine_refuses_an_unknown_parameter_slot" {
  let b = @moongrad.Binding::new()
  let raised = try {
    let _ = @moongrad.Dual::from_param(b, 0, 0)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "the_dual_engine_refuses_an_index_outside_the_parameter" {
  let b = @moongrad.Binding::new()
  b.set_param(0, [1.0])
  let raised = try {
    let _ = @moongrad.Dual::from_param(b, 0, 5)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "the_forward_mode_reads_the_value_of_a_relu_derivative_node" {
  // The dual engine reported the value of a ReluDeriv node as
  // `value * 0.0`, which is zero on both sides of the kink. The
  // node's value is the step function itself.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let d = g.relu_deriv(x)
  let (above_value, above_tangent) = @moongrad.forward_mode(
    g,
    audit_binding_of([2.0]),
    d,
    0,
  )
  assert_eq(above_value, 1.0)
  assert_eq(above_tangent, 0.0)
  let (below_value, _below_tangent) = @moongrad.forward_mode(
    g,
    audit_binding_of([-2.0]),
    d,
    0,
  )
  assert_eq(below_value, 0.0)
}

///|
test "the_forward_mode_reads_a_relu_derivative_at_an_infinity" {
  // The dual engine computed the value of a ReluDeriv node as
  // `value * 0.0`, which is a NaN when the operand is an infinity —
  // and an infinity is what a saturated activation produces. The
  // node's value is the step function, and its tangent is zero.
  let g = @moongrad.Graph::new()
  let inf = g.constant(1.0 / 0.0)
  let d = g.relu_deriv(inf)
  let b = @moongrad.Binding::new()
  let (value, tangent) = @moongrad.forward_mode(g, b, d, 0)
  assert_eq(value, 1.0)
  assert_eq(tangent, 0.0)
  // The reverse-mode forward pass has always answered one here; the
  // two have to agree.
  assert_eq(@moongrad.forward(g, b, d).output, 1.0)
}

// ---------------------------------------------------------------------------
// The gradient program
// ---------------------------------------------------------------------------

///|
test "a_relu_gradient_is_correct_wherever_the_program_is_evaluated" {
  // The documentation used to say that a ReLU's gradient was frozen
  // at the point it was compiled. It is not: the branch is taken
  // where the gradient is evaluated, so the same program follows the
  // sign of a new value.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.relu(g.sub(x, g.constant(1.0)))
  let (_grads, prog) = @moongrad.grad(g, audit_binding_of([2.0]), f)
  // At x = 2 the input to the ReLU is positive.
  let above = @moongrad.evaluate_gradients(prog, audit_binding_of([2.0]))
  assert_eq(above[0], 1.0)
  // At x = 0 the same program sees a negative input, and the
  // gradient follows it.
  let below = @moongrad.evaluate_gradients(prog, audit_binding_of([0.0]))
  assert_eq(below[0], 0.0)
}

///|
test "a_square_is_a_power_node_like_any_other_power" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let sq = g.pow(x, 2.0)
  assert_eq(g.get(sq), @moongrad.Node::PowBy(2.0, x))
  let (grads, _p) = @moongrad.grad(g, audit_binding_of([3.0]), sq)
  // d/dx x^2 = 2x = 6 at x = 3.
  assert_true(grads[0] > 5.9999999 && grads[0] < 6.0000001)
}

// ---------------------------------------------------------------------------
// The optimiser
// ---------------------------------------------------------------------------

///|
test "plain_adam_applies_the_bias_correction_on_the_first_step" {
  // The documentation used to say that plain `Adam` skipped the
  // correction for callers driving a single step at a time. It does
  // not: the first update is corrected by 1 - beta^1, which makes it
  // a full-size step in the direction of the gradient.
  let p = [1.0]
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Adam(0.1), 1)
  st.step(p, [1.0])
  // m_hat = 1 and v_hat = 1, so the step is lr / (1 + eps) and the
  // parameter ends up just below 0.9. Without the correction the
  // step would be a third of that.
  assert_true(p[0] > 0.8999 && p[0] < 0.9001)
}

// ---------------------------------------------------------------------------
// Losses over no data
// ---------------------------------------------------------------------------

///|
test "a_loss_over_no_samples_is_refused" {
  let g = @moongrad.Graph::new()
  let mse_raised = try {
    let _ = @moongrad.mse_loss(g, [], [])
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(mse_raised)
  let bce_raised = try {
    let _ = @moongrad.bce_loss(g, [], [])
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(bce_raised)
}

///|
test "the_host_cross_entropy_mean_over_no_samples_is_refused" {
  let raised = try {
    let _ = @moongrad.bce_value_mean([], [])
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "a_trainer_over_no_samples_is_refused" {
  let raised = try {
    let _ = @moongrad.Trainer::new(
      [1],
      [@moongrad.Activation::Identity],
      audit_empty_dataset(),
      @moongrad.LossKind::Mse,
      @moongrad.Optimizer::Sgd(0.1),
      1,
    )
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

// ---------------------------------------------------------------------------
// The reported loss and the optimised loss
// ---------------------------------------------------------------------------

///|
test "the_host_cross_entropy_and_the_graph_agree_at_a_large_negative_z" {
  // The host-side value used to be computed with the naive
  // `ln(1 + e^-|z|)`, which rounds to zero for a large negative z
  // while the graph's softplus does not. The two are now the same
  // function, so they agree exactly.
  for z in [-50.0, -12.0, -0.5, 0.0, 3.0, 40.0] {
    for y in [0.0, 1.0] {
      let g = @moongrad.Graph::new()
      ignore(g.new_vars(1))
      let v = g.variable(0)
      let f = @moongrad.bce_loss(g, [v], [y])
      let from_graph = @moongrad.forward(g, audit_binding_of([z]), f).output
      let from_host = @moongrad.bce_value(z, y)
      assert_eq(from_graph, from_host)
    }
  }
}

///|
test "the_host_cross_entropy_mean_agrees_with_the_graph_loss" {
  // Two samples whose whole loss is the softplus term, so that a
  // host-side value that dropped the last digits of the log would
  // show up in the average instead of being rounded away by a
  // larger term.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let zs = [-50.0, -30.0]
  let ys = [0.0, 0.0]
  let preds : Array[Int32] = []
  for i = 0; i < 2; i = i + 1 {
    preds.push(g.variable(i))
  }
  let f = @moongrad.bce_loss(g, preds, ys)
  let from_graph = @moongrad.forward(g, audit_binding_of(zs), f).output
  let from_host = @moongrad.bce_value_mean(zs, ys)
  assert_eq(from_graph, from_host)
}

///|
test "the_host_mean_squared_error_agrees_with_the_graph_loss" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let xs = [1.0, -2.0]
  let ys = [0.0, 1.0]
  let preds : Array[Int32] = []
  for i = 0; i < 2; i = i + 1 {
    preds.push(g.variable(i))
  }
  let f = @moongrad.mse_loss(g, preds, ys)
  let from_graph = @moongrad.forward(g, audit_binding_of(xs), f).output
  // ((1 - 0)^2 + (-2 - 1)^2) / 2 = (1 + 9) / 2 = 5.
  assert_eq(from_graph, 5.0)
}

// ---------------------------------------------------------------------------
// The trainer's binding
// ---------------------------------------------------------------------------

///|
test "a_binding_is_refused_for_a_graph_that_cannot_hold_the_dataset" {
  // The graph is what says how many variables the forward pass will
  // read, so a binding that does not fit it is refused rather than
  // quietly filled with slots the graph never reads.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let net = @moongrad.MLP::new(g, [2, 1], [@moongrad.Activation::Identity], 1)
  let raised = try {
    let _ = @moongrad.trainer_binding(
      g,
      audit_one_sample_dataset(),
      [0.0, 0.0, 0.0],
      net,
    )
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

// ---------------------------------------------------------------------------
// The Newton step
// ---------------------------------------------------------------------------

///|
test "the_newton_step_solves_h_y_equals_g_and_returns_x_minus_y" {
  // f(x) = x^2 / 2 - 3x has its minimum at x = 3. The step solves
  // H y = g and returns x - y, which lands on the minimum in one
  // step from anywhere.
  let h = @moongrad.Matrix::zeros(1, 1)
  h.set(0, 0, 1.0)
  match @moongrad.newton_step([0.0], [-3.0], h, 0.0) {
    Some(x) => assert_eq(x[0], 3.0)
    None => assert_true(false)
  }
}
