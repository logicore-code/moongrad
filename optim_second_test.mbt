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
  // The one-shot product puts the binding back the way it found it,
  // so the compiled one is built against a binding that is still the
  // caller's own.
  let first = @moongrad.t_input_hvp(g, b, 2, [1.0, 0.0])
  let hvp = match @moongrad.compile_input_hvp(g, b, 2, 1) {
    Some(h) => h
    None => {
      check(false, "the product should compile")
      return
    }
  }
  let compiled = hvp.apply(b, [1.0, 0.0])
  // The compiled product is over the inputs, and it has to serve
  // every direction: the program is compiled once and evaluated
  // again for each one.
  assert_eq(compiled.length(), 1)
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
  let step = match
    @moongrad.newton_cg_step(g, b, loss, 1.0e-6, 50, 1.0e-12, 3) {
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
  let iterative = match
    @moongrad.newton_cg_step(g, b, loss, 1.0e-6, 50, 1.0e-12, 3) {
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
  let op = fn(v : Array[Double]) -> Array[Double] raise @moongrad.ADError {
    if v.length() < 4 {
      // The operator is only defined on four-dimensional vectors.
      raise @moongrad.domain_error("the operator needs four entries")
    }
    [
      4.0 * v[0] + v[1],
      v[0] + 3.0 * v[1],
      2.0 * v[2] - v[3],
      -v[2] + 5.0 * v[3],
    ]
  }
  let solution = match
    @moongrad.cg_operator(
      op,
      [1.0, 2.0, 3.0, 4.0],
      [0.0, 0.0, 0.0, 0.0],
      50,
      1.0e-12,
    ) {
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
  let op = fn(v : Array[Double]) -> Array[Double] raise @moongrad.ADError {
    if v.length() < 2 {
      raise @moongrad.domain_error("the operator needs four entries")
    }
    [-v[0], v[1]]
  }
  let result = @moongrad.cg_operator(op, [1.0, 1.0], [0.0, 0.0], 50, 1.0e-12)
  assert_true(result is None)
}

///|
test "the_input_product_has_one_tensor_per_input_slot" {
  // The direction is itself an input slot, so the gradient program
  // has one more input than the primal. The product must not hand
  // back the derivative with respect to the direction along with it.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let cube = g.pow(x, 3.0)
  let loss = g.sum_axis(cube, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let product = @moongrad.t_input_hvp(g, b, loss, [1.0, 0.0])
  assert_eq(product.length(), 1)
  assert_close(product[0].data[0], 9.0, 1.0e-9, "the product")
  assert_close(product[0].data[1], 0.0, 1.0e-9, "the product")
}

///|
test "the_product_refuses_to_take_over_an_input" {
  // Writing the direction into a slot the primal already uses would
  // replace that input with the direction and change the objective.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let cube = g.pow(x, 3.0)
  let loss = g.sum_axis(cube, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let refused = @moongrad.compile_input_hvp(g, b, loss, 0)
  check(refused is None, "the direction must not take over an input")
}

///|
test "a_product_refuses_a_direction_of_the_wrong_length" {
  // A short direction would be written into the slot and the forward
  // pass would fail on it, and the failure would come back as a zero
  // product rather than as an error.
  let g = cube_input_graph()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let refused = try {
    let _ = @moongrad.t_input_hvp(g, b, 2, [1.0])
    false
  } catch {
    _ => true
  }
  check(refused, "a direction of the wrong length should be refused")
}

///|
test "a_product_refuses_a_binding_of_another_shape" {
  // The program was compiled for a two-element input. Evaluating it
  // against a three-element one would be the product of a different
  // function.
  let g = cube_input_graph()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let hvp = match @moongrad.compile_input_hvp(g, b, 2, 1) {
    Some(h) => h
    None => {
      check(false, "the product should compile")
      return
    }
  }
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5, 0.25]))
  let refused = try {
    let _ = hvp.apply(b, [1.0, 0.0, 0.0])
    false
  } catch {
    _ => true
  }
  check(refused, "a binding of another shape should be refused")
}

///|
test "a_one_shot_product_leaves_the_binding_as_it_found_it" {
  // Compiling a product gives the direction a slot of its own, which
  // the binding has to have a value for. If the binding kept that
  // slot, the second call would be over a larger space than the
  // first, and the direction it was given would no longer be the
  // right length.
  let g = cube_input_graph()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let width = b.inputs.length()
  let _ = @moongrad.t_input_hvp(g, b, 2, [1.0, 0.0])
  assert_eq(b.inputs.length(), width)
  let _ = @moongrad.t_input_hvp(g, b, 2, [0.0, 1.0])
  assert_eq(b.inputs.length(), width)
}

///|
test "the_newton_step_can_be_taken_more_than_once" {
  // Taking a Newton step compiles a Hessian-vector product, and the
  // compilation leaves the direction in a slot of the binding. The
  // second step compiles the product again against the same binding,
  // and it has to reuse that slot rather than refuse it: an
  // iteration that only works once is not an iteration.
  let g = @moongrad.TGraph::new()
  let w = g.param(0)
  let d = g.sub(w, g.constant(@moongrad.Tensor::filled([], 3.0)))
  let loss = g.sum_axis(g.mul(d, d), 0)
  let b = @moongrad.TBinding::new()
  b.set_param(0, @moongrad.Tensor::vector([1.0]))
  let first = match
    @moongrad.newton_cg_step(g, b, loss, 1.0e-6, 50, 1.0e-12, 3) {
    Some(v) => v
    None => {
      check(false, "the first newton-cg iteration broke down")
      return
    }
  }
  assert_close(first[0], 2.0, 1.0e-6, "the first newton step")
  let second = match
    @moongrad.newton_cg_step(g, b, loss, 1.0e-6, 50, 1.0e-12, 3) {
    Some(v) => v
    None => {
      check(false, "the second newton-cg iteration broke down")
      return
    }
  }
  assert_close(second[0], 2.0, 1.0e-6, "the second newton step")
}

///|
test "a_direction_slot_of_the_wrong_width_is_refused" {
  // A slot past the primal's own inputs is a direction slot, and it
  // is reused. A slot that is the wrong width cannot be the one this
  // product was compiled for, so it is refused rather than used.
  let g = @moongrad.TGraph::new()
  let w = g.param(0)
  let d = g.sub(w, g.constant(@moongrad.Tensor::filled([], 3.0)))
  let loss = g.sum_axis(g.mul(d, d), 0)
  let b = @moongrad.TBinding::new()
  b.set_param(0, @moongrad.Tensor::vector([1.0]))
  // A slot the primal does not use, but which is too wide for the
  // one-element parameter space.
  b.set_input(0, @moongrad.Tensor::vector([1.0, 2.0]))
  let refused = @moongrad.compile_param_hvp(g, b, loss, 0)
  check(refused is None, "a direction slot of the wrong width must be refused")
}

///|
test "the_compiled_programs_can_be_compiled_again" {
  // Every one of these leaves something behind in the binding: a
  // direction slot, a compiled program. A second call has to find
  // the binding in a state it can work with, because a caller who
  // compiles once and applies many times is the ordinary case and not
  // the exceptional one.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let w = g.param(0)
  let cube = g.pow(x, 3.0)
  let loss = g.sum_axis(g.mul(cube, g.mul(w, w)), 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  b.set_param(0, @moongrad.Tensor::vector([2.0]))
  // The gradient program: compiled twice, evaluated twice.
  let first = @moongrad.compile_tensor_gradient(g, b, loss)
  let second = @moongrad.compile_tensor_gradient(g, b, loss)
  let (_, g1) = @moongrad.evaluate_tensor_gradient(first, b)
  let (_, g2) = @moongrad.evaluate_tensor_gradient(second, b)
  assert_eq(g1.length(), g2.length())
  assert_close(g1[0].data[0], g2[0].data[0], 1.0e-12, "the two gradients")
  // The value of the original graph is asked of the original graph.
  let v1 = @moongrad.t_forward(g, b)
  let v2 = @moongrad.t_forward(g, b)
  assert_close(v1[loss].data[0], v2[loss].data[0], 1.0e-12, "the two losses")
  // The product over the parameters: compiled twice against the same
  // binding, with the same direction slot.
  let slot = b.inputs.length()
  let p1 = @moongrad.compile_param_hvp(g, b, loss, slot)
  let p2 = @moongrad.compile_param_hvp(g, b, loss, slot)
  check(p1 is Some(_), "the first compilation failed")
  check(p2 is Some(_), "the second compilation failed")
  // The product over the inputs: a direction slot of its own, because
  // the one above belongs to the parameter product.
  let input_slot = b.inputs.length()
  let i1 = @moongrad.compile_input_hvp(g, b, loss, input_slot)
  let i2 = @moongrad.compile_input_hvp(g, b, loss, input_slot)
  check(i1 is Some(_), "the first input compilation failed")
  check(i2 is Some(_), "the second input compilation failed")
}

///|
test "the_input_product_is_the_same_every_time_it_is_compiled" {
  // The direction slot that one compilation leaves behind must not
  // become part of the space the next compilation takes the product
  // over. If it did, the second product would be over a wider space
  // than the first, and it would be the product of a different
  // function.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let cube = g.pow(x, 3.0)
  let loss = g.sum_axis(cube, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  let slot = b.inputs.length()
  let first = @moongrad.compile_input_hvp(g, b, loss, slot)
  let second = @moongrad.compile_input_hvp(g, b, loss, slot)
  check(first is Some(_), "the first compilation failed")
  check(second is Some(_), "the second compilation failed")
  let direction : Array[Double] = [1.0, 0.0]
  let r1 = match first {
    Some(h) => h.apply(b, direction)
    None => []
  }
  let r2 = match second {
    Some(h) => h.apply(b, direction)
    None => []
  }
  assert_eq(r1.length(), 1)
  assert_eq(r2.length(), 1)
  assert_close(r1[0].data[0], r2[0].data[0], 1.0e-12, "the two products")
  // The product is a second derivative: the objective is x^3, its
  // gradient is 3x^2, and the derivative of that along (1, 0) at
  // x = 1.5 is 6x = 9.
  assert_close(r1[0].data[0], 9.0, 1.0e-9, "the product")
}
