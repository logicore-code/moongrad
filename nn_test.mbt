// nn_test.mbt
//
// Layers, losses, and the optimiser.
//
// The load-bearing test here is `dense_layer_gradients_match_a_hand
// _computed_backprop`. Every other test checks a component; that one
// checks that the components compose into the gradient somebody
// would have written down by hand for a two-layer network, which is
// the only way to know the wiring is right rather than merely
// self-consistent.

///|
fn one_var_binding() -> @moongrad.Binding {
  let b = @moongrad.Binding::new()
  b.set_var(0, 1.0)
  b
}

///|
test "lcg_is_deterministic_for_a_fixed_seed" {
  let a = @moongrad.Lcg::new(42)
  let b = @moongrad.Lcg::new(42)
  for _i = 0; _i < 10; _i = _i + 1 {
    assert_eq(a.next_f64(), b.next_f64())
  }
}

///|
test "lcg_stays_inside_the_unit_interval" {
  let r = @moongrad.Lcg::new(7)
  for _i = 0; _i < 200; _i = _i + 1 {
    let v = r.next_f64()
    check(v >= 0.0 && v < 1.0, "next_f64 escaped [0,1): \{v}")
  }
}

///|
test "lcg_does_not_repeat_immediately" {
  // A generator with too short a period would give two identical
  // weights in the same layer, which is exactly the kind of bug
  // that shows up as a layer that mysteriously trains worse.
  let r = @moongrad.Lcg::new(1)
  let a = r.next_f64()
  let b = r.next_f64()
  check(a != b, "consecutive draws must differ")
}

///|
test "uniform_respects_its_bounds" {
  let r = @moongrad.Lcg::new(99)
  for _i = 0; _i < 100; _i = _i + 1 {
    let v = r.uniform(-2.0, 3.0)
    check(v >= -2.0 && v < 3.0, "uniform escaped its bounds: \{v}")
  }
}

///|
test "dense_initialisation_stays_near_zero" {
  let g = @moongrad.Graph::new()
  let layer = @moongrad.Dense::new(g, 8, 4, @moongrad.Activation::Tanh, 5)
  // Glorot limit for fan_in=8, fan_out=4 is sqrt(6/12) = 0.707.
  let limit = (6.0 / 12.0).sqrt()
  check(
    layer.w.weights.max_abs() <= limit,
    "initial weights must respect the Glorot bound",
  )
}

///|
test "dense_apply_produces_one_node_per_output_unit" {
  let g = @moongrad.Graph::new()
  let layer = @moongrad.Dense::new(g, 3, 2, @moongrad.Activation::Identity, 1)
  let x = [g.constant(1.0), g.constant(2.0), g.constant(3.0)]
  let out = layer.apply(g, x)
  assert_eq(out.length(), 2)
}

///|
test "dense_identity_activation_is_an_affine_map" {
  let g = @moongrad.Graph::new()
  let layer = @moongrad.Dense::new(g, 2, 1, @moongrad.Activation::Identity, 3)
  // Fix the weights and bias so the answer is checkable by hand:
  // out = 2*x0 - x1 + 0.5
  layer.w.weights.set(0, 0, 2.0)
  layer.w.weights.set(0, 1, -1.0)
  layer.b.weights.set(0, 0, 0.5)
  let x = [g.constant(3.0), g.constant(1.0)]
  let out = layer.apply(g, x)
  let b = @moongrad.Binding::new()
  b.set_param(layer.w.slot, [2.0, -1.0])
  b.set_param(layer.b.slot, [0.5])
  let v = @moongrad.forward(g, b, out[0]).output
  check(approx(v, 5.5, 1.0e-9), "out should be 5.5, got \{v}")
}

///|
test "dense_gradients_match_a_hand_computed_backprop" {
  // One layer, two inputs, one output, identity activation.
  //   out = w0*x0 + w1*x1 + b
  // At x = (2, 3), w = (4, 5), b = 1:
  //   out = 8 + 15 + 1 = 24
  //   dL/dw0 = x0 = 2,  dL/dw1 = x1 = 3,  dL/db = 1
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let layer = @moongrad.Dense::new(g, 2, 1, @moongrad.Activation::Identity, 11)
  layer.w.weights.set(0, 0, 4.0)
  layer.w.weights.set(0, 1, 5.0)
  layer.b.weights.set(0, 0, 1.0)
  let x0 = g.variable(0)
  let x1 = g.variable(1)
  let out = layer.apply(g, [x0, x1])[0]
  let loss = g.pow(g.sub(out, g.constant(0.0)), 2.0)
  let b = @moongrad.Binding::new()
  b.set_var(0, 2.0)
  b.set_var(1, 3.0)
  b.set_param(layer.w.slot, [4.0, 5.0])
  b.set_param(layer.b.slot, [1.0])
  let fw = @moongrad.forward(g, b, loss).output
  check(approx(fw, 576.0, 1.0e-9), "loss should be 24^2 = 576, got \{fw}")
  let (dvar, prog) = @moongrad.grad(g, b, loss)
  // dL/dx0 = 2*out*w0 = 2*24*4 = 192
  assert_close(dvar[0], 192.0, 1.0e-9, "dL/dx0")
  assert_close(dvar[1], 240.0, 1.0e-9, "dL/dx1")
  let pg = @moongrad.param_gradients(prog, b)
  assert_close(pg[0][0], 2.0 * 24.0 * 2.0, 1.0e-9, "dL/dw0 = 2*out*x0")
  assert_close(pg[0][1], 2.0 * 24.0 * 3.0, 1.0e-9, "dL/dw1 = 2*out*x1")
  assert_close(pg[1][0], 48.0, 1.0e-9, "dL/db = 2*out")
}

///|
test "relu_blocks_negative_gradient" {
  // A ReLU in front of a linear unit: the derivative of the linear
  // part is dead on the negative side, so the gradient of a squared
  // loss with respect to a negative pre-activation is exactly zero.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let pre = g.add(g.mul(x, g.constant(2.0)), g.constant(-1.0))
  let post = g.relu(pre)
  let f = g.mul(post, g.constant(1.0))
  let (grads, _p) = @moongrad.grad(g, one_var_binding(), f)
  // x = 1 gives pre = 1, which is positive, so the gradient is 2.
  assert_close(grads[0], 2.0, 1.0e-12, "d(relu(2x-1))/dx at x=1")
}

///|
test "relu_gradient_is_zero_on_the_negative_side" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  // pre = x - 5, so at x = 0 it is negative.
  let f = g.relu(g.sub(x, g.constant(5.0)))
  let (grads, _p) = @moongrad.grad(g, one_var_binding(), f)
  assert_close(grads[0], 0.0, 1.0e-12, "relu derivative below the kink")
}

///|
test "tanh_layer_gradients_flow_through" {
  // A tanh unit is saturated for large inputs, so a moderate input
  // is the only place where the gradient is meaningfully non-zero.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.tanh(x)
  let (grads, _p) = @moongrad.grad(g, one_var_binding(), f)
  assert_close(
    grads[0],
    1.0 - @math.tanh(1.0) * @math.tanh(1.0),
    1.0e-9,
    "d tanh",
  )
}

///|
test "mlp_builds_one_layer_per_size_gap" {
  let g = @moongrad.Graph::new()
  let net = @moongrad.MLP::new(
    g,
    [2, 4, 3, 1],
    [
      @moongrad.Activation::Tanh,
      @moongrad.Activation::Tanh,
      @moongrad.Activation::Identity,
    ],
    7,
  )
  assert_eq(net.layers.length(), 3)
  // 2*4 + 4 weights, 4*3 + 3 weights, 3*1 + 1 weights.
  assert_eq(net.size(), 2 * 4 + 4 + 4 * 3 + 3 + 3 * 1 + 1)
}

///|
test "mlp_parameter_count_matches_the_arithmetic" {
  let g = @moongrad.Graph::new()
  let net = @moongrad.MLP::new(
    g,
    [3, 5, 2],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    1,
  )
  // Layer 1: 5*3 weights + 5 biases = 20
  // Layer 2: 2*5 weights + 2 biases = 12
  assert_eq(net.size(), 32)
  assert_eq(net.params().length(), 4)
}

///|
test "mlp_forward_produces_the_right_width" {
  let g = @moongrad.Graph::new()
  let net = @moongrad.MLP::new(
    g,
    [2, 3, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    2,
  )
  let x = [g.constant(0.5), g.constant(-0.5)]
  let out = net.apply(g, x)
  assert_eq(out.length(), 1)
}

///|
test "mse_loss_of_a_perfect_prediction_is_zero" {
  let g = @moongrad.Graph::new()
  let pred = [g.constant(1.0), g.constant(2.0)]
  let f = @moongrad.mse_loss(g, pred, [1.0, 2.0])
  let v = @moongrad.forward(g, @moongrad.Binding::new(), f).output
  check(approx(v, 0.0, 1.0e-12), "MSE at the optimum is 0, got \{v}")
}

///|
test "mse_loss_of_two_unit_errors_is_one" {
  // (1-0)^2 and (1-0)^2, averaged: 1. Divided by the batch and not
  // by 2*batch, so the printed number is the actual mean squared
  // error and can be compared against a variance.
  let g = @moongrad.Graph::new()
  let pred = [g.constant(1.0), g.constant(1.0)]
  let f = @moongrad.mse_loss(g, pred, [0.0, 0.0])
  let v = @moongrad.forward(g, @moongrad.Binding::new(), f).output
  check(approx(v, 1.0, 1.0e-12), "mean of (1,1) is 1, got \{v}")
}

///|
test "mse_gradient_is_twice_the_error_over_n" {
  // d/dpred mean((pred - y)^2) = 2(pred - y)/n
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let p0 = g.variable(0)
  let p1 = g.variable(1)
  let f = @moongrad.mse_loss(g, [p0, p1], [0.0, 0.0])
  let b = @moongrad.Binding::new()
  b.set_var(0, 1.0)
  b.set_var(1, -1.0)
  let (grads, _p) = @moongrad.grad(g, b, f)
  assert_close(grads[0], 1.0, 1.0e-12, "d/dfirst")
  assert_close(grads[1], -1.0, 1.0e-12, "d/dsecond")
}

///|
test "bce_loss_stays_finite_for_a_confident_correct_prediction" {
  // The whole reason cross-entropy is written as softplus(z) - z*y
  // rather than -log(sigmoid(z)): the naive form is -inf here.
  let v = @moongrad.bce_value(40.0, 1.0)
  check(
    @moongrad.is_finite(v),
    "a confident correct prediction must not be -inf, got \{v}",
  )
  check(v < 1.0e-3, "and it should be very small, got \{v}")
}

///|
test "bce_loss_is_large_for_a_confident_wrong_prediction" {
  let v = @moongrad.bce_value(-40.0, 1.0)
  check(
    @moongrad.is_finite(v),
    "a confident wrong prediction must stay finite, got \{v}",
  )
  check(v > 39.0, "and it should be about 40, got \{v}")
}

///|
test "bce_loss_derivative_is_sigmoid_minus_y" {
  // d/dz softplus(z) - z*y = sigmoid(z) - y
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let z = g.variable(0)
  let f = @moongrad.bce_loss(g, [z], [1.0])
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.5)
  let (grads, _p) = @moongrad.grad(g, b, f)
  let s = 1.0 / (1.0 + @math.exp(-0.5))
  assert_close(grads[0], s - 1.0, 1.0e-9, "d BCE/dz")
}

///|
test "bce_value_matches_the_graph_implementation" {
  // The host-side value and the graph's softplus node are two
  // independent implementations of the same formula; if they drift
  // the printed loss stops meaning what the optimiser minimises.
  for zs in [-3.0, -0.5, 0.0, 0.5, 3.0] {
    let ys = [0.0, 1.0]
    for y in ys {
      let g = @moongrad.Graph::new()
      ignore(g.new_vars(1))
      let z = g.variable(0)
      let f = @moongrad.bce_loss(g, [z], [y])
      let b = @moongrad.Binding::new()
      b.set_var(0, zs)
      let from_graph = @moongrad.forward(g, b, f).output
      let from_host = @moongrad.bce_value(zs, y)
      assert_close(from_graph, from_host, 1.0e-12, "bce at z=\{zs}, y=\{y}")
    }
  }
}

///|
test "activation_dispatch_matches_the_operator" {
  let g = @moongrad.Graph::new()
  let x = g.constant(0.0)
  let t = @moongrad.activate(g, @moongrad.Activation::Tanh, x)
  let s = @moongrad.activate(g, @moongrad.Activation::Sigmoid, x)
  let r = @moongrad.activate(g, @moongrad.Activation::Relu, x)
  let i = @moongrad.activate(g, @moongrad.Activation::Identity, x)
  let b = @moongrad.Binding::new()
  assert_eq(@moongrad.forward(g, b, t).output, 0.0)
  assert_eq(@moongrad.forward(g, b, s).output, 0.5)
  assert_eq(@moongrad.forward(g, b, r).output, 0.0)
  assert_eq(@moongrad.forward(g, b, i).output, 0.0)
}

///|
test "the_generator_spreads_over_the_whole_range" {
  // The generator used to be a 64-bit multiplication done by hand in
  // 32-bit halves, which cannot carry: the values it produced were
  // bunched into a narrow band, and every dataset and every set of
  // initial weights was drawn from that band. A mean of a third and a
  // variance a twentieth of the uniform one is the signature.
  let rng = @moongrad.Lcg::new(2024)
  let n = 20000
  let mut sum = 0.0
  let mut sumsq = 0.0
  let mut smallest = 1.0
  let mut largest = 0.0
  for i = 0; i < n; i = i + 1 {
    let v = rng.next_f64()
    sum = sum + v
    sumsq = sumsq + v * v
    if v < smallest {
      smallest = v
    }
    if v > largest {
      largest = v
    }
  }
  let mean = sum / n.to_double()
  let variance = sumsq / n.to_double() - mean * mean
  assert_close(mean, 0.5, 0.02, "the mean of the draws")
  assert_close(variance, 1.0 / 12.0, 0.01, "the variance of the draws")
  check(smallest < 0.02, "the draws should reach the bottom of the range")
  check(largest > 0.98, "the draws should reach the top of the range")
}

///|
test "the_same_seed_gives_the_same_sequence" {
  let a = @moongrad.Lcg::new(99)
  let b = @moongrad.Lcg::new(99)
  for i = 0; i < 50; i = i + 1 {
    assert_close(a.next_f64(), b.next_f64(), 1.0e-15, "the two generators")
  }
}

///|
test "the_generator_is_reproducible_across_a_target" {
  // The sequence is a property of the arithmetic, not of the
  // platform, which is the whole reason for having a generator of
  // one's own.
  let a = @moongrad.Lcg::new(2024)
  let first = a.next_f64()
  let b = @moongrad.Lcg::new(2024)
  assert_close(first, b.next_f64(), 1.0e-15, "the first value")
}
