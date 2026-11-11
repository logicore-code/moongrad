// tensor_program_test.mbt
//
// The gradient as a re-usable program.
//
// These tests exist because of a specific failure mode: a backward
// pass that folds each local derivative into a number is right at the
// point it was compiled and wrong everywhere else. The tests below
// compile a gradient once and then evaluate it at other inputs, so a
// frozen derivative cannot pass them.
//
// The Hessian tests are the other half: they differentiate the
// gradient program, which is only possible if the gradient really is
// a program.

///|
fn tensor_binding(vals : Array[Double]) -> @moongrad.TBinding {
  let b = @moongrad.TBinding::new()
  let t = @moongrad.Tensor::zeros([vals.length()])
  for i = 0; i < vals.length(); i = i + 1 {
    t.data[i] = vals[i]
  }
  b.set_input(0, t)
  b
}

///|
/// f(x) = sum_i relu(x_i) * x_i, so that
///   df/dx_i = 2 x_i   when x_i > 0
///           = 0       otherwise.
fn relu_square_graph() -> @moongrad.TGraph {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let r = g.relu(x)
  let y = g.mul(r, x)
  ignore(g.sum_axis(y, 0))
  g
}

///|
test "a_compiled_tensor_gradient_is_reusable_at_another_input" {
  // Compile once, at a point where every element is positive.
  let g = relu_square_graph()
  let compiled = @moongrad.compile_tensor_gradient(
    g,
    tensor_binding([1.0, 2.0, 3.0]),
    2,
  )
  // Evaluate at a point where every element is negative. A gradient
  // frozen at the compile point would still report 2x here.
  let (grads, _) = @moongrad.evaluate_tensor_gradient(
    compiled,
    tensor_binding([-1.0, -2.0, -3.0]),
  )
  let d = grads[0]
  assert_close(d.data[0], 0.0, 1.0e-12, "df/dx0 at a negative input")
  assert_close(d.data[1], 0.0, 1.0e-12, "df/dx1 at a negative input")
  assert_close(d.data[2], 0.0, 1.0e-12, "df/dx2 at a negative input")
}

///|
test "a_compiled_tensor_gradient_re_evaluates_across_the_relu_kink" {
  let g = relu_square_graph()
  let compiled = @moongrad.compile_tensor_gradient(
    g,
    tensor_binding([1.0, -1.0, 1.0]),
    2,
  )
  let (grads, _) = @moongrad.evaluate_tensor_gradient(
    compiled,
    tensor_binding([-2.0, 3.0, 0.5]),
  )
  let d = grads[0]
  // x0 was positive at the compile point and is negative now, so the
  // gradient has to drop from 2 to 0.  x1 was negative and is
  // positive, so it has to rise from 0 to 6.  x2 was positive and is
  // still positive, so it has to follow the input down to 1.
  assert_close(d.data[0], 0.0, 1.0e-12, "df/dx0, now negative")
  assert_close(d.data[1], 6.0, 1.0e-12, "df/dx1, now positive")
  assert_close(d.data[2], 1.0, 1.0e-12, "df/dx2, now positive")
}

///|
test "a_compiled_tensor_gradient_agrees_with_a_freshly_compiled_one" {
  let g = relu_square_graph()
  let b = tensor_binding([0.5, -0.5, 2.0])
  let compiled = @moongrad.compile_tensor_gradient(g, b, 2)
  let (reused, _) = @moongrad.evaluate_tensor_gradient(compiled, b)
  let fresh = @moongrad.t_grad(g, b, 2)
  for i = 0; i < 3; i = i + 1 {
    assert_close(
      reused[0].data[i],
      fresh.input_grads[0].data[i],
      1.0e-12,
      "reused gradient",
    )
  }
}

///|
test "the_tensor_hessian_matches_the_closed_form" {
  // f(x) = sum_i x_i^3, so d2f/dx_i dx_j = 6 x_i when i = j.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let cube = g.pow(x, 3.0)
  ignore(g.sum_axis(cube, 0))
  let h = @moongrad.t_hessian(g, tensor_binding([1.0, 2.0]), 1)
  assert_close(h.get(0, 0), 6.0, 1.0e-9, "d2f/dx0^2")
  assert_close(h.get(0, 1), 0.0, 1.0e-9, "d2f/dx0 dx1")
  assert_close(h.get(1, 0), 0.0, 1.0e-9, "d2f/dx1 dx0")
  assert_close(h.get(1, 1), 12.0, 1.0e-9, "d2f/dx1^2")
}

///|
test "the_tensor_hessian_is_symmetric" {
  // f(x, y) = x * y^2 has d2f/dxdy = 2y and d2f/dydx = 2y, so the
  // off-diagonal entries have to agree even though the two sides are
  // derived by different routes.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let y = g.variable(1)
  let f = g.mul(x, g.pow(y, 2.0))
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::filled([], 3.0))
  b.set_input(1, @moongrad.Tensor::filled([], 5.0))
  let h = @moongrad.t_hessian(g, b, f)
  assert_close(h.get(0, 1), 10.0, 1.0e-9, "d2f/dxdy")
  assert_close(h.get(1, 0), 10.0, 1.0e-9, "d2f/dydx")
}

///|
test "the_tensor_hessian_of_a_mixed_product_has_the_right_off_diagonals" {
  // f(x, y) = x^2 * y.  d2f/dx2 = 2y, d2f/dxy = 2x, d2f/dy2 = 0.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let y = g.variable(1)
  let f = g.mul(g.pow(x, 2.0), y)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::filled([], 3.0))
  b.set_input(1, @moongrad.Tensor::filled([], 5.0))
  let h = @moongrad.t_hessian(g, b, f)
  assert_close(h.get(0, 0), 10.0, 1.0e-9, "d2f/dx2 = 2y")
  assert_close(h.get(0, 1), 6.0, 1.0e-9, "d2f/dxdy = 2x")
  assert_close(h.get(1, 0), 6.0, 1.0e-9, "d2f/dydx = 2x")
  assert_close(h.get(1, 1), 0.0, 1.0e-9, "d2f/dy2 = 0")
}

///|
test "the_tensor_hessian_of_a_relu_is_zero_away_from_the_kink" {
  // f(x) = relu(x)^2.  For x > 0 the function is x^2, so the second
  // derivative is 2; for x < 0 it is 0, so the second derivative is
  // 0 as well. The kink itself is where the second derivative does
  // not exist.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let r = g.relu(x)
  let f = g.sum_axis(g.pow(r, 2.0), 0)
  let h = @moongrad.t_hessian(g, tensor_binding([2.0, -2.0]), f)
  assert_close(h.get(0, 0), 2.0, 1.0e-9, "d2f/dx0^2 above the kink")
  assert_close(h.get(1, 1), 0.0, 1.0e-9, "d2f/dx1^2 below the kink")
}

///|
test "the_tensor_hessian_matches_finite_differences_of_the_gradient" {
  // The gradient of sum_i x_i^3 is 3 x_i^2, so the Hessian is the
  // derivative of that: 6 x_i.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let cube = g.pow(x, 3.0)
  ignore(g.sum_axis(cube, 0))
  let h = @moongrad.t_hessian(g, tensor_binding([1.5, -0.5]), 1)
  let eps = 1.0e-5
  // The central difference of the gradient at x_i.
  for i = 0; i < 2; i = i + 1 {
    let up = [1.5, -0.5]
    let down = [1.5, -0.5]
    up[i] = up[i] + eps
    down[i] = down[i] - eps
    let gu = @moongrad.t_grad(g, tensor_binding(up), 1)
    let gd = @moongrad.t_grad(g, tensor_binding(down), 1)
    let numeric = (gu.input_grads[0].data[i] - gd.input_grads[0].data[i]) /
      (2.0 * eps)
    assert_close(h.get(i, i), numeric, 1.0e-5, "hessian against differences")
  }
}

///|
test "the_tensor_parameter_hessian_matches_the_closed_form" {
  // f(w, x) = w * x^2 with w a scalar parameter.  d2f/dw2 = 0,
  // d2f/dwdx = 2x, d2f/dx2 = 2w.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let w = g.param(0)
  let f = g.mul(w, g.pow(x, 2.0))
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::filled([], 3.0))
  b.set_param(0, @moongrad.Tensor::filled([], 5.0))
  let h = @moongrad.t_param_hessian(g, b, f)
  assert_close(h.get(0, 0), 0.0, 1.0e-9, "d2f/dw2")
}

///|
test "the_mixed_tensor_hessian_crosses_parameters_and_inputs" {
  // f(w, x) = w * x^2 with w a scalar parameter and x a scalar
  // input.  d2f/dwdx = 2x and d2f/dxdw = 2x, so the mixed block has
  // to agree with the input-space Hessian's off-diagonal.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let w = g.param(0)
  let f = g.mul(w, g.pow(x, 2.0))
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::filled([], 3.0))
  b.set_param(0, @moongrad.Tensor::filled([], 5.0))
  let h = @moongrad.t_mixed_hessian(g, b, f)
  assert_close(h.get(0, 0), 6.0, 1.0e-9, "d2f/dwdx = 2x")
}
