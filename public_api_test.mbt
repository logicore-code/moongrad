// public_api_test.mbt
//
// The public functions nothing else in the repository calls.
//
// Every function here is part of the library's surface: it is
// exported, documented, and meant to be used. The rest of the suite
// happens not to use them, which means they were shipped without
// ever having been run.
//
// The tests below are short — one call each, checked against a value
// worked out by hand. They are here so that "it is exported" and "it
// works" are the same statement.
//
// The host-side elementwise operators are checked against the closed
// form of each one; the reductions against a value worked out on
// paper; the derivative helpers against the scalar engine's table,
// which is the table they wrap.

///|
/// A two-row matrix.
fn two_by_two() -> @moongrad.Tensor {
  match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
    Some(t) => t
    None => @moongrad.Tensor::zeros([2, 2])
  }
}

///|
test "the_host_side_elementwise_operators" {
  let x = @moongrad.Tensor::vector([0.5, 1.5, 2.5])
  let y = @moongrad.Tensor::vector([1.0, 2.0, 3.0])
  assert_close(@moongrad.t_neg(x).data[0], -0.5, 1.0e-12, "neg")
  assert_close(@moongrad.t_exp(x).data[0], 1.6487212707001282, 1.0e-12, "exp")
  assert_close(@moongrad.t_log(x).data[0], @math.ln(0.5), 1.0e-12, "log")
  assert_close(@moongrad.t_sqrt(x).data[0], 0.7071067811865476, 1.0e-12, "sqrt")
  assert_close(@moongrad.t_sin(x).data[0], @math.sin(0.5), 1.0e-12, "sin")
  assert_close(@moongrad.t_cos(x).data[0], @math.cos(0.5), 1.0e-12, "cos")
  assert_close(@moongrad.t_tanh(x).data[0], @math.tanh(0.5), 1.0e-12, "tanh")
  assert_close(
    @moongrad.t_sigmoid(x).data[0],
    1.0 / (1.0 + @math.exp(-0.5)),
    1.0e-12,
    "sigmoid",
  )
  assert_close(
    @moongrad.t_softplus(x).data[0],
    @math.ln(1.0 + @math.exp(0.5)),
    1.0e-12,
    "softplus",
  )
  assert_close(@moongrad.t_relu(x).data[0], 0.5, 1.0e-12, "relu")
  assert_close(
    @moongrad.t_relu(@moongrad.Tensor::vector([-1.0])).data[0],
    0.0,
    1.0e-12,
    "relu of a negative",
  )
  assert_close(@moongrad.t_shift(x, 1.0).data[0], 1.5, 1.0e-12, "shift")
  let sum = match @moongrad.t_add(x, y) {
    Some(t) => t
    None => return
  }
  assert_close(sum.data[0], 1.5, 1.0e-12, "add")
  let diff = match @moongrad.t_sub(x, y) {
    Some(t) => t
    None => return
  }
  assert_close(diff.data[0], -0.5, 1.0e-12, "sub")
  let quot = match @moongrad.t_div(x, y) {
    Some(t) => t
    None => return
  }
  assert_close(quot.data[0], 0.5, 1.0e-12, "div")
  let prod = match @moongrad.t_mul(x, y) {
    Some(t) => t
    None => return
  }
  assert_close(prod.data[0], 0.5, 1.0e-12, "mul")
}

///|
test "the_host_side_reductions" {
  let x = @moongrad.Tensor::vector([1.0, 2.0, 3.0, 4.0])
  assert_close(@moongrad.t_sum(x), 10.0, 1.0e-12, "the sum")
  assert_close(@moongrad.t_mean(x), 2.5, 1.0e-12, "the mean")
  let m = match @moongrad.t_matmul(two_by_two(), two_by_two()) {
    Some(t) => t
    None => return
  }
  // [[1,2],[3,4]] squared is [[7,10],[15,22]].
  assert_close(m.data[0], 7.0, 1.0e-12, "the matrix product")
  assert_close(m.data[3], 22.0, 1.0e-12, "the matrix product")
  let joined = match @moongrad.t_concat(x, x, 0) {
    Some(t) => t
    None => return
  }
  assert_eq(joined.shape.length(), 1)
  assert_eq(joined.data.length(), 8)
  assert_close(joined.data[4], 1.0, 1.0e-12, "the concatenation")
  let transposed = match @moongrad.t_transpose(two_by_two()) {
    Some(t) => t
    None => return
  }
  assert_close(transposed.data[1], 3.0, 1.0e-12, "the transpose")
  let mse = match
    @moongrad.t_mse(
      @moongrad.Tensor::vector([1.0, 2.0]),
      @moongrad.Tensor::vector([0.0, 0.0]),
    ) {
    Some(v) => v
    None => return
  }
  assert_close(mse, 2.5, 1.0e-12, "the mean squared error")
}

///|
test "the_host_side_predicates" {
  check(
    @moongrad.t_is_zero(@moongrad.Tensor::zeros([3])),
    "a zero tensor should be zero",
  )
  check(
    !@moongrad.t_is_zero(@moongrad.Tensor::vector([0.0, 1.0])),
    "a tensor with a one in it should not be zero",
  )
}

///|
test "the_tensor_side_derivative_helpers_agree_with_the_scalar_table" {
  // These wrap the scalar engine's table, which is the table the
  // reverse pass and the forward-mode cross-check consult.
  let x = @moongrad.Tensor::vector([0.7])
  let y = @moongrad.Tensor::vector([1.3])
  let common = [1]
  let dl = match
    @moongrad.t_binary_dleft(@moongrad.BinaryOp::Mul, x, y, common) {
    Some(t) => t
    None => return
  }
  // d(a*b)/da is b.
  assert_close(dl.data[0], 1.3, 1.0e-12, "d(a*b)/da")
  let dr = match
    @moongrad.t_binary_dright(@moongrad.BinaryOp::Mul, x, y, common) {
    Some(t) => t
    None => return
  }
  // d(a*b)/db is a.
  assert_close(dr.data[0], 0.7, 1.0e-12, "d(a*b)/db")
  let du = @moongrad.t_unary_derivative(@moongrad.UnaryOp::Tanh, x)
  // The table's form for tanh keeps the digits a cancellation would
  // lose.
  let e = @math.exp(-2.0 * 0.7)
  assert_close(
    du.data[0],
    4.0 * e / ((1.0 + e) * (1.0 + e)),
    1.0e-12,
    "the tangent's derivative",
  )
}

///|
test "the_checked_forward_evaluation_refuses_a_broken_graph" {
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.0, 2.0]))
  // A graph whose node refers to an input that was never set is
  // fine; one whose operation cannot be carried out is not.
  let ok = @moongrad.t_forward_checked(g, b)
  let ok = match ok {
    Some(_) => true
    None => false
  }
  check(ok, "a graph with no nodes should evaluate")
  // A broadcast that cannot be carried out.
  let a = g.input(0)
  let c = g.input(1)
  // Three columns cannot be broadcast against two.
  b.set_input(
    1,
    match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
      Some(t) => t
      None => return
    },
  )
  ignore(g.add(a, c))
  let broken = @moongrad.t_forward_checked(g, b)
  let refused = match broken {
    Some(_) => false
    None => true
  }
  check(refused, "a graph whose broadcast cannot be carried out should refuse")
}

///|
test "the_training_helpers" {
  // The largest gradient magnitude and the flattened parameter list.
  let grads = [
    @moongrad.Tensor::vector([0.1, -0.4]),
    @moongrad.Tensor::vector([0.2, 0.05]),
  ]
  assert_close(
    @moongrad.grad_max_abs(grads),
    0.4,
    1.0e-12,
    "the largest gradient",
  )
  let flat = @moongrad.flatten_params(grads)
  assert_eq(flat.length(), 4)
  assert_close(flat[1], -0.4, 1.0e-12, "the flattened parameter")
}

///|
test "the_scalar_engine_helpers" {
  // The adjoint of a slot, and the number of parameters a graph
  // declares.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let y = g.mul(x, x)
  let b = @moongrad.Binding::new()
  b.set_var(0, 3.0)
  // eval_adjoint evaluates a node of the *gradient* program, which
  // holds the primal below the gradient arithmetic, so evaluating it
  // evaluates both.
  let compiled = @moongrad.compile_gradient(g, y)
  let adj = @moongrad.eval_adjoint(compiled.graph, b, compiled.var_adj[0])
  assert_close(adj, 6.0, 1.0e-12, "the adjoint of x^2 at 3")
  let with_params = @moongrad.Graph::new()
  ignore(with_params.new_vars(0))
  let slot = with_params.new_param(2)
  ignore(slot)
  assert_eq(@moongrad.total_params(with_params), 2)
}

///|
test "the_normalisation_mean_helper" {
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let m = @moongrad.mean_over(g, x, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([2.0, 4.0]))
  let values = @moongrad.t_forward(g, b)
  assert_close(values[m].data[0], 3.0, 1.0e-12, "the mean")
}

///|
test "the_cropping_helper" {
  // t_crop2d removes the padding a convolution added.
  let x = match
    @moongrad.Tensor::from_data([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0]) {
    Some(t) => t
    None => return
  }
  let padded = @moongrad.t_pad2d(x, 1, 1)
  let cropped = @moongrad.t_crop2d(padded, 1, 1)
  assert_eq(cropped.shape.length(), 4)
  assert_eq(cropped.data.length(), 4)
  assert_close(cropped.data[0], 1.0, 1.0e-12, "the first value after cropping")
}
