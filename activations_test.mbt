// activations_test.mbt
//
// The new activations.
//
// Both are checked the same way: against central differences of
// their own forward pass, in both engines. The finite difference is
// an independent derivation, so agreement is evidence rather than a
// tautology.

///|
fn scalar_binding(v : Double) -> @moongrad.Binding {
  let b = @moongrad.Binding::new()
  b.set_var(0, v)
  b
}

///|
test "the_scalar_silu_matches_its_definition" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let y = g.silu(x)
  let b = scalar_binding(0.7)
  let value = @moongrad.forward(g, b, y).output
  let expected = 0.7 / (1.0 + @math.exp(-0.7))
  assert_close(value, expected, 1.0e-12, "silu")
}

///|
test "the_scalar_gelu_matches_its_definition" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let y = g.gelu(x)
  let b = scalar_binding(0.7)
  let value = @moongrad.forward(g, b, y).output
  let c = 0.7978845608028654
  let u = c * (0.7 + 0.044715 * 0.7 * 0.7 * 0.7)
  let expected = 0.5 * 0.7 * (1.0 + @math.tanh(u))
  assert_close(value, expected, 1.0e-12, "gelu")
}

///|
test "the_scalar_silu_gradient_matches_finite_differences" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let y = g.silu(x)
  for v in [-2.0, -0.3, 0.0, 0.4, 1.7] {
    let (grads, _p) = @moongrad.grad(g, scalar_binding(v), y)
    let h = 1.0e-6
    let up = scalar_binding(v + h)
    let down = scalar_binding(v - h)
    let numeric = (
        @moongrad.forward(g, up, y).output -
        @moongrad.forward(g, down, y).output
      ) /
      (2.0 * h)
    assert_close(grads[0], numeric, 1.0e-5, "silu gradient")
  }
}

///|
test "the_scalar_gelu_gradient_matches_finite_differences" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let y = g.gelu(x)
  for v in [-2.0, -0.3, 0.0, 0.4, 1.7] {
    let (grads, _p) = @moongrad.grad(g, scalar_binding(v), y)
    let h = 1.0e-6
    let up = scalar_binding(v + h)
    let down = scalar_binding(v - h)
    let numeric = (
        @moongrad.forward(g, up, y).output -
        @moongrad.forward(g, down, y).output
      ) /
      (2.0 * h)
    assert_close(grads[0], numeric, 1.0e-5, "gelu gradient")
  }
}

///|
test "the_tensor_silu_matches_its_definition" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let y = g.silu(x)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([0.7, -0.2]))
  let values = @moongrad.t_forward(g, b)
  assert_close(
    values[y].data[0],
    0.7 / (1.0 + @math.exp(-0.7)),
    1.0e-12,
    "silu",
  )
  assert_close(
    values[y].data[1],
    -0.2 / (1.0 + @math.exp(0.2)),
    1.0e-12,
    "silu",
  )
}

///|
test "the_tensor_gelu_gradient_matches_finite_differences" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let y = g.gelu(x)
  let loss = g.sum_axis(y, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([0.7, -0.2]))
  let grads = @moongrad.t_grad(g, b, loss)
  let h = 1.0e-6
  for i = 0; i < 2; i = i + 1 {
    let up = @moongrad.Tensor::vector([0.7, -0.2])
    let down = @moongrad.Tensor::vector([0.7, -0.2])
    up.data[i] = up.data[i] + h
    down.data[i] = down.data[i] - h
    let bu = @moongrad.TBinding::new()
    bu.set_input(0, up)
    let bd = @moongrad.TBinding::new()
    bd.set_input(0, down)
    let vu = @moongrad.t_forward(g, bu)
    let vd = @moongrad.t_forward(g, bd)
    let numeric = (vu[loss].data[0] - vd[loss].data[0]) / (2.0 * h)
    assert_close(grads.input_grads[0].data[i], numeric, 1.0e-5, "gelu gradient")
  }
}
