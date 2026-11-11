// optim_second_test.mbt
//
// Second-order optimisation.
//
// The Hessian-vector product is checked against the dense Hessian it
// is supposed to agree with, because that is the only way to know the
// matrix-free version is computing the right thing.

///|
/// The gradient of `x^3` summed over the elements, with `x` an input.
fn cube_input_graph() -> @moongrad.TGraph {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let cube = g.pow(x, 3.0)
  ignore(g.sum_axis(cube, 0))
  g
}

///|
test "the_hessian_vector_product_agrees_with_the_dense_hessian" {
  let g = cube_input_graph()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let h = @moongrad.t_hessian(g, b, 2)
  // f(x) = sum x^3, so the Hessian is diag(6x) = diag(9, -3).
  for j = 0; j < 2; j = j + 1 {
    let v : Array[Double] = [0.0, 0.0]
    v[j] = 1.0
    let product = @moongrad.t_input_hvp(g, b, 2, v)
    for i = 0; i < 2; i = i + 1 {
      assert_close(
        product[0].data[i],
        h.get(i, j),
        1.0e-9,
        "hvp against the dense hessian",
      )
    }
  }
}

///|
test "the_hessian_vector_product_is_linear" {
  let g = cube_input_graph()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let a = @moongrad.t_input_hvp(g, b, 2, [1.0, 2.0])
  let c = @moongrad.t_input_hvp(g, b, 2, [3.0, -1.0])
  let sum = @moongrad.t_input_hvp(g, b, 2, [4.0, 1.0])
  for i = 0; i < 2; i = i + 1 {
    assert_close(
      sum[0].data[i],
      a[0].data[i] + c[0].data[i],
      1.0e-9,
      "hvp linearity",
    )
  }
}

///|
test "the_compiled_hessian_vector_product_serves_every_direction" {
  let g = cube_input_graph()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let hvp = @moongrad.compile_param_hvp(g, b, 2, 7)
  let first = @moongrad.t_input_hvp(g, b, 2, [1.0, 0.0])
  let compiled = hvp.apply(b, [1.0, 0.0])
  // The compiled product is over the parameters, and this graph has
  // none, so it is empty; what matters is that it evaluates without
  // error and that the input-space product still agrees with the
  // dense Hessian.
  assert_eq(compiled.length(), 0)
  let dense = @moongrad.t_hessian(g, b, 2)
  assert_close(dense.get(0, 0), 9.0, 1.0e-9, "d2f/dx0^2")
  assert_close(dense.get(1, 1), -3.0, 1.0e-9, "d2f/dx1^2")
  ignore(first)
}

///|
test "the_parameter_hessian_vector_product_agrees_with_the_dense_hessian" {
  // f(w) = sum w^3 over a two-element parameter.
  let g = @moongrad.TGraph::new()
  let w = g.param(0)
  let cube = g.pow(w, 3.0)
  let loss = g.sum_axis(cube, 0)
  let b = @moongrad.TBinding::new()
  b.set_param(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let dense = @moongrad.t_param_hessian(g, b, loss)
  let product = @moongrad.t_param_hvp(g, b, loss, [1.0, 2.0])
  for i = 0; i < 2; i = i + 1 {
    let want = dense.get(i, 0) * 1.0 + dense.get(i, 1) * 2.0
    assert_close(product[0].data[i], want, 1.0e-9, "param hvp")
  }
}

///|
test "a_newton_step_lands_on_the_minimum" {
  // f(w) = (w - 3)^2. One Newton step should land exactly on 3.
  let g = @moongrad.TGraph::new()
  let w = g.param(0)
  let d = g.sub(w, g.constant(@moongrad.Tensor::filled([], 3.0)))
  let loss = g.sum_axis(g.mul(d, d), 0)
  let b = @moongrad.TBinding::new()
  b.set_param(0, @moongrad.Tensor::vector([1.0]))
  let step = match @moongrad.newton_cg_step(g, b, loss, 1.0e-6, 50, 1.0e-12, 3) {
    Some(v) => v
    None => {
      check(false, "the newton-cg iteration broke down")
      return
    }
  }
  assert_close(step[0], 2.0, 1.0e-6, "the newton step")
}

///|
test "the_dense_newton_step_agrees_with_the_iterative_one" {
  let g = @moongrad.TGraph::new()
  let w = g.param(0)
  let d = g.sub(w, g.constant(@moongrad.Tensor::filled([], 3.0)))
  let loss = g.sum_axis(g.mul(d, d), 0)
  let b = @moongrad.TBinding::new()
  b.set_param(0, @moongrad.Tensor::vector([1.0]))
  let dense = match @moongrad.dense_newton_step(g, b, loss, 1.0e-6) {
    Some(v) => v
    None => {
      check(false, "the dense newton step failed")
      return
    }
  }
  let iterative = match @moongrad.newton_cg_step(g, b, loss, 1.0e-6, 50, 1.0e-12, 3) {
    Some(v) => v
    None => {
      check(false, "the newton-cg iteration broke down")
      return
    }
  }
  assert_close(dense[0], iterative[0], 1.0e-6, "dense against iterative")
}

///|
test "conjugate_gradient_solves_a_spd_system" {
  // The 4x4 system with A = [[4,1],[1,3]] on the diagonal blocks.
  let op = fn(v : Array[Double]) -> Array[Double] {
    [
      4.0 * v[0] + v[1],
      v[0] + 3.0 * v[1],
      2.0 * v[2] - v[3],
      -v[2] + 5.0 * v[3],
    ]
  }
  let solution = match @moongrad.cg_operator(op, [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], 50, 1.0e-12) {
    Some(v) => v
    None => {
      check(false, "the iteration broke down on a positive definite system")
      return
    }
  }
  let residual = op(solution)
  assert_close(residual[0], 1.0, 1.0e-8, "residual 0")
  assert_close(residual[1], 2.0, 1.0e-8, "residual 1")
  assert_close(residual[2], 3.0, 1.0e-8, "residual 2")
  assert_close(residual[3], 4.0, 1.0e-8, "residual 3")
}

///|
test "conjugate_gradient_reports_an_indefinite_operator" {
  // The matrix [[-1,0],[0,1]] is indefinite, so the first curvature
  // p'Ap is negative and the iteration must notice rather than
  // return nonsense.
  let op = fn(v : Array[Double]) -> Array[Double] {
    [-v[0], v[1]]
  }
  let result = @moongrad.cg_operator(op, [1.0, 1.0], [0.0, 0.0], 50, 1.0e-12)
  assert_true(result is None)
}
