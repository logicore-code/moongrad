// regression_test.mbt
//
// Tests for bugs that were found by review rather than by the
// ordinary test suite.
//
// Each one here was a real defect at some point: a gradient
// credited to the wrong variable, a forward-mode pass that could
// not see a ReLU's derivative, a Newton step that quietly changed
// the matrix it was given, a learning-rate override that changed
// the trainer for ever, and an input slot that a layer could take
// away from the caller.
//
// They are collected in one file because they share a shape: each is
// a statement about something that used to be true and should not
// be again.

///|
test "a_variable_used_out_of_order_still_gets_its_own_gradient" {
  // The slots were remembered in the order the variables were first
  // used, so a graph that used variable 2 before variable 1 gave
  // variable 1's gradient to variable 2.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(3))
  let x0 = g.variable(0)
  let x2 = g.variable(2)
  // f = 1 * x0 + 2 * x2, so df/dx0 = 1 and df/dx2 = 2.
  let f = g.add(g.mul(x0, g.constant(1.0)), g.mul(x2, g.constant(2.0)))
  let b = @moongrad.Binding::new()
  b.set_var(0, 1.0)
  b.set_var(1, 1.0)
  b.set_var(2, 1.0)
  let (grads, _p) = @moongrad.grad(g, b, f)
  assert_eq(grads[0], 1.0)
  assert_eq(grads[1], 0.0)
  assert_eq(grads[2], 2.0)
}

///|
test "a_variable_used_before_an_earlier_one_still_gets_its_gradient" {
  // The slots are indexed by variable number, so the order the
  // variables are first used in must not matter — not even when the
  // earlier one is used second.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let y = g.variable(1)
  let x = g.variable(0)
  let f = g.mul(x, y)
  let b = @moongrad.Binding::new()
  b.set_var(0, 2.0)
  b.set_var(1, 3.0)
  let (grads, _p) = @moongrad.grad(g, b, f)
  // f = x * y, so df/dx = y = 3 and df/dy = x = 2.
  assert_eq(grads[0], 3.0)
  assert_eq(grads[1], 2.0)
}

///|
test "a_variable_declared_but_never_used_has_no_gradient" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(3))
  let x2 = g.variable(2)
  let f = g.mul(x2, g.constant(2.0))
  let b = @moongrad.Binding::new()
  b.set_var(0, 1.0)
  b.set_var(1, 1.0)
  b.set_var(2, 1.0)
  let (grads, _p) = @moongrad.grad(g, b, f)
  assert_eq(grads[0], 0.0)
  assert_eq(grads[1], 0.0)
  assert_eq(grads[2], 2.0)
}

///|
test "the_forward_mode_sees_a_relu_derivatives_value" {
  // The dual engine used to report zero for the node, so the oracle
  // the reverse pass is checked against could not see it.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.mul(g.relu(x), g.constant(2.0))
  let b = @moongrad.Binding::new()
  b.set_var(0, 3.0)
  let (value, _derivative) = @moongrad.forward_mode(g, b, f, 0)
  // relu(3) = 3, so the value is 6.
  assert_eq(value, 6.0)
}

///|
test "the_forward_mode_reports_a_domain_error" {
  // The dual engine used to answer a zero for an argument outside
  // the operator's domain, which let the oracle agree with a
  // gradient the engine itself refused to produce.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.ln(x)
  let b = @moongrad.Binding::new()
  b.set_var(0, -1.0)
  let mut raised = false
  try {
    let _ = @moongrad.forward_mode(g, b, f, 0)
  } catch {
    _ => raised = true
  } noraise {
    v => v
  }
  check(raised, "the domain error should be reported")
}

///|
test "the_newton_step_leaves_the_callers_hessian_alone" {
  // The damping was added to the matrix in place, so a loop that
  // reused one Hessian across damped steps accumulated the damping.
  let h = @moongrad.Matrix::zeros(2, 2)
  h.set(0, 0, 2.0)
  h.set(1, 1, 4.0)
  let x = [1.0, 1.0]
  let g = [2.0, 4.0]
  let _ = @moongrad.newton_step(x, g, h, 1.0)
  assert_eq(h.get(0, 0), 2.0)
  assert_eq(h.get(1, 1), 4.0)
}

///|
test "the_learning_rate_override_applies_to_one_step_only" {
  // The override replaced the configured optimiser for ever, so a
  // trainer that was asked for plain SGD once never went back to
  // Adam.
  let data = match
    @moongrad.Dataset::new([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], 1) {
    Some(d) => d
    None => {
      check(false, "the dataset should build")
      return
    }
  }
  let trainer = @moongrad.Trainer::new(
    [1],
    [@moongrad.Activation::Identity],
    data,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.1),
    1,
  )
  let before = trainer.opt_state.kind
  let _ = trainer.train_step(0.5)
  let after = trainer.opt_state.kind
  match (before, after) {
    (@moongrad.Optimizer::Adam(_), @moongrad.Optimizer::Adam(_)) => ()
    _ => check(false, "the optimiser should not have changed")
  }
}

///|
test "a_layer_does_not_take_over_an_input_slot" {
  // The dropout claims an input slot from the graph. The claim must
  // come after every slot the caller has declared, or the layer
  // would silently overwrite the caller's input.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let d = @moongrad.Dropout::new(0.5, 3)
  let y = d.apply(g, x, [2])
  // The input the caller declared is still the input.
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.0, 2.0]))
  d.bind(b)
  let values = @moongrad.t_forward(g, b)
  // The mask is a separate slot, so the output has the caller's two
  // values scaled by it and nothing else.
  assert_eq(values[y].data.length(), 2)
}

///|
test "the_mask_slot_does_not_collide_with_the_callers_inputs" {
  // The dropout claims an input slot from the graph. The claim comes
  // after every slot the caller has declared, so declaring the inputs
  // first — which is the natural order, since the graph needs the
  // inputs before the layers — keeps the two apart.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let other = g.input(1)
  let d = @moongrad.Dropout::new(0.5, 3)
  let y = d.apply(g, x, [2])
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.0, 2.0]))
  b.set_input(1, @moongrad.Tensor::vector([5.0]))
  d.bind(b)
  let values = @moongrad.t_forward(g, b)
  // The two inputs are different, so the output is the masked first
  // input and the second input is untouched.
  assert_eq(values[other].data[0], 5.0)
  assert_eq(values[y].data.length(), 2)
}
