// optim_extra_test.mbt
//
// The extra optimisers and the training utilities around them,
// checked against the formulas written in the module header rather
// than against themselves.

///|
/// A one-element parameter tensor.
fn one_param(x : Double) -> @moongrad.Tensor {
  @moongrad.Tensor::vector([x])
}

///|
/// The loss of the quadratic `0.5 * x^2` at `x`. Its gradient is `x`.
fn quad_loss(x : Double) -> Double {
  0.5 * x * x
}

///|
test "rmsprop_converges_on_a_quadratic" {
  // The gradient of 0.5*x^2 is x, so this is gradient descent with a
  // step size that RMSProp fixes from the running average of the
  // squared gradient. The loss starts at 12.5 and ends three orders
  // of magnitude lower.
  let params = [one_param(5.0)]
  let st = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::RmsProp(0.05, 0.9),
    [[1]],
  )
  let start_loss = quad_loss(params[0].data[0])
  for i = 0; i < 300; i = i + 1 {
    let g = [one_param(params[0].data[0])]
    st.step(params, g, i + 1)
  }
  let loss = quad_loss(params[0].data[0])
  check(
    loss < 1.0e-3,
    "RMSProp should drive the loss to near zero, got \{loss}",
  )
  check(
    loss < start_loss / 100.0,
    "RMSProp should cut the loss by two orders of magnitude, got \{loss}",
  )
}

///|
test "adagrad_converges_on_a_quadratic" {
  // Adagrad's accumulator grows with the squares of the gradients it
  // has seen, so the effective step shrinks as the run goes on. That
  // is what brings the quadratic down to nothing rather than to a
  // limit cycle.
  let params = [one_param(5.0)]
  let st = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::Adagrad(0.5, 1.0e-8),
    [[1]],
  )
  let start_loss = quad_loss(params[0].data[0])
  for i = 0; i < 300; i = i + 1 {
    let g = [one_param(params[0].data[0])]
    st.step(params, g, i + 1)
  }
  let loss = quad_loss(params[0].data[0])
  check(
    loss < 1.0e-6,
    "Adagrad should drive the loss to near zero, got \{loss}",
  )
  check(
    loss < start_loss / 100.0,
    "Adagrad should cut the loss by two orders of magnitude, got \{loss}",
  )
}

///|
test "nesterov_updates_by_the_expected_formula" {
  // The velocity after the first step is mu*0 + g = 1, so the update
  // is lr*(g + mu*v) = 0.1*(1 + 0.9) = 0.19, and 2.0 becomes 1.81.
  let params = [one_param(2.0)]
  let st = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::Nesterov(0.1, 0.9),
    [[1]],
  )
  st.step(params, [one_param(1.0)], 1)
  assert_close(params[0].data[0], 1.81, 1.0e-12, "Nesterov after one step")
  // The velocity carries over: it is now mu*1 + 1 = 1.9, so the
  // second step is 0.1*(1 + 0.9*1.9) = 0.271 and 1.81 becomes 1.539.
  st.step(params, [one_param(1.0)], 2)
  assert_close(params[0].data[0], 1.539, 1.0e-12, "Nesterov after two steps")
}

///|
test "adamw_decays_the_parameter_even_with_a_zero_gradient" {
  // With a zero gradient the adaptive part of the update is zero, so
  // the only thing left is the decay: p -= lr*wd*p = 0.1*0.5*1.0, and
  // 1.0 becomes 0.95. This is the point of the decoupled form — the
  // parameter keeps shrinking towards zero even when the loss says
  // nothing about it.
  let params = [one_param(1.0)]
  let st = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::AdamW(0.1, 0.5, 1.0e-8),
    [[1]],
  )
  st.step(params, [one_param(0.0)], 1)
  assert_close(params[0].data[0], 0.95, 1.0e-12, "AdamW with a zero gradient")
}

///|
test "adamw_and_the_coupled_form_give_different_results" {
  // The coupled form folds the decay into the gradient before Adam
  // ever sees it, so the gradient becomes g + wd*p = 0.5 and the
  // decay is then scaled by the running standard deviation like any
  // other gradient. On the first step that standard deviation is
  // 0.5 itself, so the step is a full lr and 1.0 becomes 0.9.
  let coupled_params = [one_param(1.0)]
  let coupled_state = @moongrad.TensorOptState::new(
    @moongrad.TensorOptimizer::Adam(0.1),
    [[1]],
  )
  coupled_state.step(coupled_params, [one_param(0.5)], 1)
  assert_close(
    coupled_params[0].data[0],
    0.9,
    1.0e-8,
    "Adam with the decay folded into the gradient",
  )
  // The decoupled form leaves the gradient alone and adds lr*wd*p to
  // the step, so the parameter shrinks by a fifth rather than a
  // tenth.
  let params = [one_param(1.0)]
  let st = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::AdamW(0.1, 0.5, 1.0e-8),
    [[1]],
  )
  st.step(params, [one_param(0.0)], 1)
  assert_close(
    params[0].data[0],
    0.95,
    1.0e-12,
    "AdamW with the decay kept separate",
  )
  check(
    @moongrad.absf(coupled_params[0].data[0] - params[0].data[0]) > 0.04,
    "the coupled and decoupled forms should disagree: \{coupled_params[0].data[0]} vs \{params[0].data[0]}",
  )
}

///|
test "clip_grad_norm_returns_the_norm_before_clipping" {
  // The gradient is (3, 4), so its norm is 5. Clipping to 2.5 scales
  // it by a half, which leaves the norm exactly at the limit and the
  // direction of the gradient untouched.
  let grads = [@moongrad.Tensor::vector([3.0, 4.0])]
  let norm = @moongrad.clip_grad_norm(grads, 2.5)
  assert_eq(norm, 5.0)
  assert_close(
    @moongrad.grad_norm_tensors(grads),
    2.5,
    1.0e-12,
    "the norm after clipping",
  )
  assert_close(grads[0].data[0], 1.5, 1.0e-12, "the first entry after clipping")
  assert_close(
    grads[0].data[1],
    2.0,
    1.0e-12,
    "the second entry after clipping",
  )
}

///|
test "clip_grad_norm_leaves_a_zero_gradient_alone" {
  // A zero gradient has no direction to preserve, so there is
  // nothing to scale — and dividing by its norm would give a NaN.
  let grads = [@moongrad.Tensor::vector([0.0, 0.0])]
  let norm = @moongrad.clip_grad_norm(grads, 1.0)
  assert_eq(norm, 0.0)
  assert_eq(grads[0].data[0], 0.0)
  assert_eq(grads[0].data[1], 0.0)
}

///|
test "clip_grad_norm_leaves_a_gradient_inside_the_limit_alone" {
  // The norm is 0.5, which is inside the limit of 1.0, so the
  // gradient is returned untouched.
  let grads = [@moongrad.Tensor::vector([0.3, 0.4])]
  let norm = @moongrad.clip_grad_norm(grads, 1.0)
  assert_close(norm, 0.5, 1.0e-12, "the norm before clipping")
  assert_close(grads[0].data[0], 0.3, 1.0e-12, "the first entry is untouched")
  assert_close(grads[0].data[1], 0.4, 1.0e-12, "the second entry is untouched")
}

///|
test "clip_grad_value_clamps_both_tails" {
  // The two tails are pulled in to the limit and the entry between
  // them is left where it is.
  let grads = [@moongrad.Tensor::vector([-5.0, 0.5, 7.0])]
  @moongrad.clip_grad_value(grads, 1.0)
  assert_eq(grads[0].data[0], -1.0)
  assert_eq(grads[0].data[1], 0.5)
  assert_eq(grads[0].data[2], 1.0)
}

///|
test "lr_schedule_constant" {
  let schedule = @moongrad.LrSchedule::Constant(0.1)
  assert_eq(schedule.at(0), 0.1)
  assert_eq(schedule.at(100), 0.1)
}

///|
test "lr_schedule_step_decays_at_the_boundary" {
  // The rate is halved every ten steps: unchanged up to the
  // boundary, halved on it.
  let schedule = @moongrad.LrSchedule::Step(0.1, 0.5, 10)
  assert_close(schedule.at(0), 0.1, 1.0e-12, "at the first step")
  assert_close(schedule.at(9), 0.1, 1.0e-12, "just before the boundary")
  assert_close(schedule.at(10), 0.05, 1.0e-12, "at the boundary")
  assert_close(schedule.at(20), 0.025, 1.0e-12, "at the second boundary")
}

///|
test "lr_schedule_exponential" {
  // The rate is halved at every single step.
  let schedule = @moongrad.LrSchedule::Exponential(0.1, 0.5)
  assert_close(schedule.at(0), 0.1, 1.0e-12, "at the first step")
  assert_close(schedule.at(1), 0.05, 1.0e-12, "after one step")
  assert_close(schedule.at(3), 0.0125, 1.0e-12, "after three steps")
}

///|
test "lr_schedule_cosine" {
  // The cosine runs from lr down to min_lr over the period and then
  // stays there: halfway along it is halfway between the two.
  let schedule = @moongrad.LrSchedule::Cosine(0.1, 0.01, 100)
  assert_close(schedule.at(0), 0.1, 1.0e-12, "at the start of the period")
  assert_close(schedule.at(50), 0.055, 1.0e-12, "at the midpoint of the period")
  assert_close(schedule.at(100), 0.01, 1.0e-12, "at the end of the period")
  assert_close(schedule.at(200), 0.01, 1.0e-12, "past the end of the period")
}

///|
/// A tensor with a given shape and one entry per position.
fn shaped(shape : Array[Int32], values : Array[Double]) -> @moongrad.Tensor {
  let t = @moongrad.Tensor::zeros(shape)
  for i = 0; i < values.length(); i = i + 1 {
    t.data[i] = values[i]
  }
  t
}

///|
test "the_optimiser_state_sizes_a_parameter_by_its_elements" {
  // A 2x3 parameter has six elements, not five. Sizing the state by
  // the sum of the shape would have two parameters share a moment
  // slot and the state run out before the parameters did.
  let state = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::AdamW(0.1, 0.0, 1.0e-8),
    [[2, 3]],
  )
  let p = [shaped([2, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]
  let g = [shaped([2, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]
  state.step(p, g, 1)
  // The update is the same for every entry, so they all move
  // together; the point is that the sixth entry did not run past the
  // end of the state.
  assert_close(p[0].data[5], 0.9, 1.0e-6, "the sixth element")
}

///|
test "two_parameters_do_not_share_a_moment_slot" {
  // The second parameter's first element must not inherit the first
  // parameter's moment.
  let state = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::AdamW(0.1, 0.0, 1.0e-8),
    [[2, 3], [2]],
  )
  let p0 = shaped([2, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  let p1 = shaped([2], [1.0, 1.0])
  let g0 = shaped([2, 3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
  let g1 = shaped([2], [1.0, 1.0])
  state.step([p0, p1], [g0, g1], 1)
  // The second parameter sees a gradient of one and nothing else,
  // so it takes the first step's full move.
  assert_close(p1.data[0], 0.9, 1.0e-6, "the second parameter")
  assert_close(p1.data[1], 0.9, 1.0e-6, "the second parameter")
}

///|
test "the_first_adamw_step_is_not_a_nan" {
  // The bias correction divides by 1 - beta^t, so t has to be at
  // least one. A zero index used to turn every parameter into a NaN.
  let state = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::AdamW(0.1, 0.5, 1.0e-8),
    [[1]],
  )
  let p = [shaped([1], [1.0])]
  let g = [shaped([1], [1.0])]
  state.step(p, g, 0)
  check(p[0].data[0] == p[0].data[0], "the parameter should not be a NaN")
}

///|
test "one_nan_does_not_poison_the_whole_gradient" {
  let g0 = shaped([2], [3.0, 4.0])
  let g1 = shaped([1], [0.0 / 0.0])
  let norm = @moongrad.clip_grad_norm([g0, g1], 1.0)
  check(norm != norm, "the norm should report that it is not finite")
  // The finite gradient is left alone.
  assert_close(g0.data[0], 3.0, 1.0e-12, "the finite gradient")
  assert_close(g0.data[1], 4.0, 1.0e-12, "the finite gradient")
}

///|
test "a_negative_clip_limit_is_a_magnitude" {
  let g = shaped([2], [5.0, -5.0])
  @moongrad.clip_grad_value([g], -1.0)
  assert_close(g.data[0], 1.0, 1.0e-12, "the clipped value")
  assert_close(g.data[1], -1.0, 1.0e-12, "the clipped value")
}

///|
test "the_optimiser_stops_at_the_end_of_its_state" {
  // A state built for one parameter, handed two, must not read past
  // its own moment arrays.
  let state = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::Nesterov(0.1, 0.0),
    [[1]],
  )
  let p = [shaped([1], [1.0]), shaped([1], [1.0])]
  let g = [shaped([1], [1.0]), shaped([1], [1.0])]
  state.step(p, g, 1)
  assert_close(p[0].data[0], 0.9, 1.0e-12, "the first parameter")
}

///|
test "a_gradient_too_large_to_square_is_scaled_rather_than_zeroed" {
  // The sum of the squares of these entries overflows, so a norm
  // computed from it would be infinite and the scale would be zero.
  let g = shaped([2], [1.0e200, 1.0e200])
  let norm = @moongrad.clip_grad_norm([g], 1.0)
  check(norm == norm, "the norm should be finite")
  let after = g.data[0] * g.data[0] + g.data[1] * g.data[1]
  check(after <= 2.0, "the scaled gradient should have a bounded length")
}
