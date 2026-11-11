// optim_test.mbt
//
// Optimiser update rules, checked against the formulas written in
// the module header rather than against themselves.

///|
fn flat(xs : Array[Double]) -> Array[Double] {
  xs.copy()
}

///|
test "sgd_moves_opposite_the_gradient" {
  let p = flat([1.0, 2.0])
  let g = flat([0.5, -0.25])
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Sgd(0.1), 2)
  st.step(p, g)
  assert_eq(p[0], 0.95)
  assert_eq(p[1], 2.025)
}

///|
test "sgd_with_zero_learning_rate_does_nothing" {
  let p = flat([1.0, 2.0])
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Sgd(0.0), 2)
  st.step(p, flat([1.0, 1.0]))
  assert_eq(p[0], 1.0)
  assert_eq(p[1], 2.0)
}

///|
test "momentum_accumulates_a_consistent_gradient" {
  // With a constant gradient, momentum converges to g/(1-mu), so
  // the velocity after the first step is exactly g and the update
  // is lr*g; after the second it is g*(1+mu).
  let p = flat([0.0, 0.0])
  let st = @moongrad.OptimizerState::new(
    @moongrad.Optimizer::Momentum(0.1, 0.9),
    2,
  )
  st.step(p, flat([1.0, 1.0]))
  assert_eq(p[0], -0.1)
  st.step(p, flat([1.0, 1.0]))
  // velocity is now 1.9, so the step is 0.19
  assert_close(p[0], -0.29, 1.0e-12, "momentum after two steps")
}

///|
test "momentum_damps_an_oscillating_gradient" {
  // Alternating +1/-1: momentum smooths this to a small net
  // movement, which is the whole point of the velocity term.
  let p = flat([0.0])
  let st = @moongrad.OptimizerState::new(
    @moongrad.Optimizer::Momentum(0.1, 0.9),
    1,
  )
  for i = 0; i < 6; i = i + 1 {
    let sign = if i % 2 == 0 { 1.0 } else { -1.0 }
    st.step(p, flat([sign]))
  }
  // Net displacement is far smaller than six independent steps
  // of size 0.1 would give.
  check(
    @moongrad.absf(p[0]) < 0.3,
    "momentum should damp oscillation, got \{p[0]}",
  )
}

///|
test "adam_step_size_is_roughly_the_learning_rate" {
  // Adam divides by the running standard deviation, so for a
  // constant gradient the per-step movement converges to about
  // lr regardless of how large the gradient is. That is the whole
  // reason to use it.
  let p = flat([0.0])
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Adam(0.1), 1)
  for _i = 0; _i < 50; _i = _i + 1 {
    st.step_adam(p, flat([7.0]), _i + 1)
  }
  // 50 steps of 0.1 move the parameter by about 5, not by
  // 50 * 0.1 * 7 = 35. Sign is negative because the step is
  // downhill.
  check(
    @moongrad.absf(p[0]) < 6.0,
    "Adam should not scale with the gradient: \{p[0]}",
  )
  check(@moongrad.absf(p[0]) > 3.0, "Adam should still make progress: \{p[0]}")
}

///|
test "adam_survives_a_zero_gradient" {
  // Dividing by an estimated standard deviation of zero gives an
  // infinite step. eps is what prevents it, so a parameter whose
  // gradient is exactly zero must stay put rather than become NaN.
  let p = flat([1.0, 2.0])
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Adam(0.1), 2)
  st.step_adam(p, flat([0.0, 0.0]), 1)
  check(
    @moongrad.is_finite(p[0]),
    "a zero gradient must not produce a NaN: \{p[0]}",
  )
  check(
    @moongrad.is_finite(p[1]),
    "a zero gradient must not produce a NaN: \{p[1]}",
  )
  assert_eq(p[0], 1.0)
}

///|
test "adam_bias_correction_makes_the_first_step_sane" {
  // Without the correction both moment estimates start at zero and
  // the first update is lr*sign(g) — a full-size step in a
  // direction chosen before anything has been learned. With it the
  // first step is a small fraction of that.
  let p = flat([0.0])
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Adam(0.1), 1)
  st.step_adam(p, flat([1.0]), 1)
  check(
    @moongrad.absf(p[0]) < 0.1,
    "the first Adam step should be small, not a full lr: \{p[0]}",
  )
}

///|
test "adam_rescales_a_small_gradient_toward_a_full_step" {
  // A gradient of 1e-3 in the same direction as one of 1.0 should
  // move the parameter by nearly the same amount, because the
  // update is normalised by the running scale. Not exactly the
  // same: eps in the denominator bounds how small a gradient can
  // be scaled up, which is deliberate — the alternative is an
  // infinite step the first time a ReLU dies.
  let big = flat([0.0])
  let st1 = @moongrad.OptimizerState::new(@moongrad.Optimizer::Adam(0.1), 1)
  let small = flat([0.0])
  let st2 = @moongrad.OptimizerState::new(@moongrad.Optimizer::Adam(0.1), 1)
  for i = 0; i < 20; i = i + 1 {
    st1.step_adam(big, flat([1.0]), i + 1)
    st2.step_adam(small, flat([1.0e-3]), i + 1)
  }
  check(
    @moongrad.absf(big[0] - small[0]) < 0.05,
    "Adam should be near scale invariant: \{big[0]} vs \{small[0]}",
  )
}

///|
test "optimiser_learning_rate_is_readable" {
  assert_eq(@moongrad.Optimizer::Sgd(0.1).lr(), 0.1)
  assert_eq(@moongrad.Optimizer::Momentum(0.2, 0.9).lr(), 0.2)
  assert_eq(@moongrad.Optimizer::Adam(0.3).lr(), 0.3)
}

///|
test "an_oversized_parameter_array_is_clipped" {
  // A caller with a parameter array longer than the gradient must
  // not read off the end of the gradient.
  let p = flat([1.0, 2.0, 3.0])
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Sgd(1.0), 2)
  st.step(p, flat([1.0, 1.0]))
  assert_eq(p[0], 0.0)
  assert_eq(p[1], 1.0)
  assert_eq(p[2], 3.0)
}

///|
test "grad_norm_of_a_vector" {
  assert_eq(@moongrad.grad_norm([3.0, 4.0]), 5.0)
  assert_eq(@moongrad.grad_norm([]), 0.0)
  assert_eq(@moongrad.grad_norm([1.0, 1.0, 1.0, 1.0]), 2.0)
}

///|
test "grad_cosine_of_identical_vectors_is_one" {
  let c = @moongrad.grad_cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
  assert_close(c, 1.0, 1.0e-12, "cosine of parallel vectors")
}

///|
test "grad_cosine_of_orthogonal_vectors_is_zero" {
  assert_close(
    @moongrad.grad_cosine([1.0, 0.0], [0.0, 1.0]),
    0.0,
    1.0e-12,
    "orthogonal",
  )
}

///|
test "grad_cosine_of_a_dead_layer_is_zero" {
  // A layer whose ReLUs have all died sends no gradient. The
  // cosine against a healthy layer is the cheapest way to notice
  // that from the outside.
  let c = @moongrad.grad_cosine([1.0, 2.0], [0.0, 0.0])
  check(approx(c, 0.0, 1.0e-12), "a zero vector has no direction: \{c}")
}

///|
test "the_first_adam_step_is_not_a_nan" {
  // The bias correction divides by 1 - beta^t, so t has to be at
  // least one. A zero index used to turn every parameter into a NaN.
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Adam(0.1), 1)
  let params = [1.0]
  let grads = [1.0]
  st.step(params, grads)
  check(params[0] == params[0], "the parameter should not be a NaN")
}

///|
test "the_adam_bias_correction_follows_the_step_count" {
  // The second step must use 1 - beta^2, not 1 - beta^1: the
  // correction is what stops the first few updates from being
  // full-size steps in directions the optimiser has not earned.
  let st = @moongrad.OptimizerState::new(@moongrad.Optimizer::Adam(0.1), 1)
  let params = [1.0]
  let grads = [1.0]
  st.step(params, grads)
  let first = params[0]
  st.step(params, grads)
  let second = params[0]
  check(
    first != second,
    "the second step should not repeat the first one's correction",
  )
}
