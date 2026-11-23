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
  let mean = match @moongrad.t_mean(x) {
    Some(v) => v
    None => return
  }
  assert_close(mean, 2.5, 1.0e-12, "the mean")
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

///|
test "the_matrix_norm_is_the_euclidean_norm" {
  let m = match @moongrad.Matrix::from_array(2, 2, [3.0, 0.0, 0.0, 4.0]) {
    Some(x) => x
    None => return
  }
  check(approx(m.norm(), 5.0, 1.0e-12), "the norm of a 3-4 matrix is 5")
}

///|
test "the_scalar_operator_table_answers_for_its_operators" {
  // The table is the numerical answer the forward-mode oracle checks
  // against, and the simplifier's identity rules read from it. These
  // are the functions the rest of the suite reached only through the
  // engine, so a wrong entry in it could go unnoticed.
  let exp = match @moongrad.apply_unary(@moongrad.UnaryOp::Exp, 0.0) {
    Some(v) => v
    None => {
      check(false, "exp(0) should exist")
      0.0
    }
  }
  check(approx(exp, 1.0, 1.0e-12), "exp(0) is 1")
  let ln = match @moongrad.apply_unary(@moongrad.UnaryOp::Ln, 1.0) {
    Some(v) => v
    None => {
      check(false, "ln(1) should exist")
      0.0
    }
  }
  check(approx(ln, 0.0, 1.0e-12), "ln(1) is 0")
  // A logarithm of a negative number does not exist, and the table
  // says so rather than returning a NaN.
  let bad = @moongrad.apply_unary(@moongrad.UnaryOp::Ln, -1.0)
  check(bad is None, "ln of a negative number does not exist")
  let sum = match @moongrad.apply_binary(@moongrad.BinaryOp::Add, 2.0, 3.0) {
    Some(v) => v
    None => {
      check(false, "2 + 3 should exist")
      0.0
    }
  }
  check(approx(sum, 5.0, 1.0e-12), "2 + 3 is 5")
  let quotient = match
    @moongrad.apply_binary(@moongrad.BinaryOp::Div, 6.0, 3.0) {
    Some(v) => v
    None => {
      check(false, "6 / 3 should exist")
      0.0
    }
  }
  check(approx(quotient, 2.0, 1.0e-12), "6 / 3 is 2")
  // A division by zero is not a number, and the table refuses it.
  let division = @moongrad.apply_binary(@moongrad.BinaryOp::Div, 1.0, 0.0)
  check(division is None, "a division by zero does not exist")
}

///|
test "the_local_derivative_table_answers_for_its_operators" {
  // These are the derivatives the forward-mode oracle uses. They have
  // to agree with the backward rules, and the agreement is checked
  // for the operators the engine reaches; the table itself is checked
  // here.
  check(
    approx(
      @moongrad.unary_derivative(@moongrad.UnaryOp::Neg, 3.0),
      -1.0,
      1.0e-12,
    ),
    "d(-x)",
  )
  check(
    approx(
      @moongrad.unary_derivative(@moongrad.UnaryOp::Exp, 0.5),
      1.6487212707001282,
      1.0e-12,
    ),
    "d(exp(x)) is exp(x)",
  )
  check(
    approx(
      @moongrad.unary_derivative(@moongrad.UnaryOp::Sqrt, 4.0),
      0.25,
      1.0e-12,
    ),
    "d(sqrt(x))",
  )
  check(
    approx(
      @moongrad.binary_dleft(@moongrad.BinaryOp::Mul, 2.0, 3.0),
      3.0,
      1.0e-12,
    ),
    "d(a*b)/da is b",
  )
  check(
    approx(
      @moongrad.binary_dright(@moongrad.BinaryOp::Mul, 2.0, 3.0),
      2.0,
      1.0e-12,
    ),
    "d(a*b)/db is a",
  )
  check(
    approx(
      @moongrad.binary_dright(@moongrad.BinaryOp::Sub, 2.0, 3.0),
      -1.0,
      1.0e-12,
    ),
    "d(a-b)/db is -1",
  )
  check(
    approx(
      @moongrad.binary_dleft(@moongrad.BinaryOp::Div, 2.0, 4.0),
      0.25,
      1.0e-12,
    ),
    "d(a/b)/da is 1/b",
  )
  check(
    approx(
      @moongrad.binary_dright(@moongrad.BinaryOp::Div, 2.0, 4.0),
      -0.125,
      1.0e-12,
    ),
    "d(a/b)/db is -a/b^2",
  )
}

///|
test "the_identity_table_answers_for_its_operators" {
  // The simplifier rewrites a node when one of its operands is the
  // identity of the operator. A literal that is wrongly taken for an
  // identity turns a computation into a different one.
  check(
    @moongrad.binary_left_identity(@moongrad.BinaryOp::Add, 0.0),
    "0 is the left identity of +",
  )
  check(
    @moongrad.binary_left_identity(@moongrad.BinaryOp::Mul, 1.0),
    "1 is the left identity of *",
  )
  check(
    !@moongrad.binary_left_identity(@moongrad.BinaryOp::Sub, 0.0),
    "0 is not the left identity of -",
  )
  check(
    !@moongrad.binary_left_identity(@moongrad.BinaryOp::Div, 1.0),
    "1 is not the left identity of /",
  )
  check(
    @moongrad.binary_right_identity(@moongrad.BinaryOp::Add, 0.0),
    "0 is the right identity of +",
  )
  check(
    @moongrad.binary_right_identity(@moongrad.BinaryOp::Sub, 0.0),
    "0 is the right identity of -",
  )
  check(
    @moongrad.binary_right_identity(@moongrad.BinaryOp::Mul, 1.0),
    "1 is the right identity of *",
  )
  check(
    @moongrad.binary_right_identity(@moongrad.BinaryOp::Div, 1.0),
    "1 is the right identity of /",
  )
}

///|
test "the_small_numeric_helpers_answer" {
  // A handful of small functions that the library uses and that no
  // other test called.
  check(approx(@moongrad.maxf(2.0, 3.0), 3.0, 1.0e-15), "maxf takes the larger")
  check(
    approx(@moongrad.maxf(-2.0, -3.0), -2.0, 1.0e-15),
    "maxf takes the larger",
  )
  check(approx(@moongrad.log1p(0.0), 0.0, 1.0e-15), "log1p(0) is 0")
  check(
    approx(@moongrad.log1p(1.0), 0.6931471805599453, 1.0e-12),
    "log1p(1) is ln 2",
  )
  check(
    approx(@moongrad.softplus_value(0.0), 0.6931471805599453, 1.0e-12),
    "softplus(0) is ln 2",
  )
  check(
    approx(@moongrad.softplus_value(-100.0), 0.0, 1.0e-15),
    "softplus of a large negative is 0",
  )
  let square = match @moongrad.safe_pow(2.0, 3.0) {
    Some(v) => v
    None => {
      check(false, "2^3 should exist")
      0.0
    }
  }
  check(approx(square, 8.0, 1.0e-12), "2^3 is 8")
  // Zero to a negative power is not a number, and the guard says so.
  let impossible = @moongrad.safe_pow(0.0, -1.0)
  check(impossible is None, "zero to a negative power does not exist")
  check(
    approx(@moongrad.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0, 1.0e-12),
    "the dot product",
  )
}

///|
test "the_number_formatting_answers" {
  // The two formatters differ in what they promise: one rounds to a
  // readable number of digits, the other prints every digit it has.
  let rounded = @moongrad.format_double(1.0 / 3.0)
  let exact = @moongrad.format_double_exact(1.0 / 3.0)
  check(rounded != exact, "the two formatters should not agree here")
  check(
    exact.contains("0.3333333333333333"),
    "the exact formatter should print the digits it has: \{exact}",
  )
  check(
    @moongrad.format_double(0.5).contains("0.5"),
    "a number that needs no rounding should be printed as it is",
  )
}

///|
test "the_mean_squared_error_of_one_prediction_is_the_squared_difference" {
  // `(pred - y)^2` for one scalar prediction.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let loss = @moongrad.mse_one(g, x, 3.0)
  let b = @moongrad.Binding::new()
  b.set_var(0, 2.0)
  // (2 - 3)^2 = 1.
  assert_close(
    @moongrad.forward(g, b, loss).output,
    1.0,
    1.0e-12,
    "the squared error",
  )
  b.set_var(0, 5.0)
  // (5 - 3)^2 = 4.
  assert_close(
    @moongrad.forward(g, b, loss).output,
    4.0,
    1.0e-12,
    "the squared error further out",
  )
}

///|
test "the_binary_cross_entropy_of_one_prediction_is_softplus_minus_z_times_y" {
  // `softplus(z) - z*y` for one scalar prediction.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let z = g.variable(0)
  let loss = @moongrad.bce_one(g, z, 1.0)
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.0)
  // softplus(0) - 0 * 1 = ln 2.
  assert_close(
    @moongrad.forward(g, b, loss).output,
    @math.ln(2.0),
    1.0e-12,
    "the cross entropy at zero",
  )
  b.set_var(0, 1.0)
  // softplus(1) - 1 * 1 = ln(1 + e) - 1.
  assert_close(
    @moongrad.forward(g, b, loss).output,
    @math.ln(1.0 + @math.exp(1.0)) - 1.0,
    1.0e-12,
    "the cross entropy at one",
  )
}

///|
test "the_broadcast_shape_is_the_shape_the_source_reaches" {
  // A row of three against a 2x3 matrix reaches the matrix's shape.
  let row = @moongrad.broadcast_shape([3], [2, 3])
  match row {
    Some(shape) => {
      assert_eq(shape.length(), 2)
      assert_eq(shape[0], 2)
      assert_eq(shape[1], 3)
    }
    None => check(false, "a row should broadcast against a matrix")
  }
  // A singleton axis takes the other one's length.
  let single = @moongrad.broadcast_shape([1, 3], [2, 3])
  match single {
    Some(shape) => {
      assert_eq(shape.length(), 2)
      assert_eq(shape[0], 2)
      assert_eq(shape[1], 3)
    }
    None => check(false, "a singleton axis should broadcast")
  }
  // Two columns against three do not broadcast.
  check(
    @moongrad.broadcast_shape([2, 3], [3, 3]) is None,
    "two columns should not broadcast against three",
  )
}

///|
test "the_compiled_hessian_vector_product_refuses_a_primal_input_slot" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let cube = g.pow(x, 3.0)
  let loss = g.sum_axis(cube, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.5, -0.5]))
  // Writing the direction over one of the primal's own inputs would
  // change the objective, so it is refused.
  let refused = @moongrad.compile_hvp(g, b, loss, 0, false)
  check(refused is None, "the product should refuse a primal input slot")
  // A slot past the primal's own inputs is a direction slot.
  let compiled = @moongrad.compile_hvp(g, b, loss, 1, false)
  let product = match compiled {
    Some(h) => h.apply(b, [1.0, 0.0])
    None => {
      check(false, "the product should compile over the inputs")
      return
    }
  }
  // f(x) = sum x^3, so the Hessian is diag(6x) = diag(9, -3), and
  // the product with (1, 0) is the first column.
  assert_eq(product.length(), 1)
  assert_close(product[0].data[0], 9.0, 1.0e-9, "the first component")
  assert_close(product[0].data[1], 0.0, 1.0e-9, "the second component")
}

///|
test "the_unary_derivative_expression_is_the_local_derivative" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.5)
  // d(exp(x)) is exp(x).
  let d_exp = @moongrad.derivative_expr(g, @moongrad.UnaryOp::Exp, x)
  assert_close(
    @moongrad.forward(g, b, d_exp).output,
    @math.exp(0.5),
    1.0e-12,
    "d(exp(x))",
  )
  // d(ln(x)) is 1/x.
  let d_ln = @moongrad.derivative_expr(g, @moongrad.UnaryOp::Ln, x)
  assert_close(@moongrad.forward(g, b, d_ln).output, 2.0, 1.0e-12, "d(ln(x))")
  // The derivative of a negation does not depend on x.
  let d_neg = @moongrad.derivative_expr(g, @moongrad.UnaryOp::Neg, x)
  assert_close(@moongrad.forward(g, b, d_neg).output, -1.0, 1.0e-12, "d(-x)")
  // d(tanh(x)) is 1 - tanh(x)^2.
  let d_tanh = @moongrad.derivative_expr(g, @moongrad.UnaryOp::Tanh, x)
  let t = @math.tanh(0.5)
  assert_close(
    @moongrad.forward(g, b, d_tanh).output,
    1.0 - t * t,
    1.0e-12,
    "d(tanh(x))",
  )
}

///|
test "the_left_derivative_is_the_partial_with_respect_to_the_first_operand" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let b = @moongrad.Binding::new()
  b.set_var(0, 4.0)
  // d(a+b)/da and d(a-b)/da are one.
  let d_add = @moongrad.derivative_left(g, @moongrad.BinaryOp::Add, x)
  assert_close(@moongrad.forward(g, b, d_add).output, 1.0, 1.0e-12, "d(a+b)/da")
  let d_sub = @moongrad.derivative_left(g, @moongrad.BinaryOp::Sub, x)
  assert_close(@moongrad.forward(g, b, d_sub).output, 1.0, 1.0e-12, "d(a-b)/da")
  // d(a*b)/da is b.
  let d_mul = @moongrad.derivative_left(g, @moongrad.BinaryOp::Mul, x)
  assert_close(@moongrad.forward(g, b, d_mul).output, 4.0, 1.0e-12, "d(a*b)/da")
  // d(a/b)/da is 1/b.
  let d_div = @moongrad.derivative_left(g, @moongrad.BinaryOp::Div, x)
  assert_close(
    @moongrad.forward(g, b, d_div).output,
    0.25,
    1.0e-12,
    "d(a/b)/da",
  )
}

///|
test "the_right_derivative_is_the_partial_with_respect_to_the_second_operand" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let a = g.variable(0)
  let c = g.variable(1)
  let b = @moongrad.Binding::new()
  b.set_var(0, 2.0)
  b.set_var(1, 4.0)
  // d(a+b)/db is one.
  let d_add = @moongrad.derivative_right(g, @moongrad.BinaryOp::Add, a, c)
  assert_close(@moongrad.forward(g, b, d_add).output, 1.0, 1.0e-12, "d(a+b)/db")
  // d(a-b)/db is minus one.
  let d_sub = @moongrad.derivative_right(g, @moongrad.BinaryOp::Sub, a, c)
  assert_close(
    @moongrad.forward(g, b, d_sub).output,
    -1.0,
    1.0e-12,
    "d(a-b)/db",
  )
  // d(a*b)/db is a.
  let d_mul = @moongrad.derivative_right(g, @moongrad.BinaryOp::Mul, a, c)
  assert_close(@moongrad.forward(g, b, d_mul).output, 2.0, 1.0e-12, "d(a*b)/db")
  // d(a/b)/db is -a/b^2.
  let d_div = @moongrad.derivative_right(g, @moongrad.BinaryOp::Div, a, c)
  assert_close(
    @moongrad.forward(g, b, d_div).output,
    -0.125,
    1.0e-12,
    "d(a/b)/db",
  )
}

///|
test "the_flat_parameter_gradients_are_in_slot_then_index_order" {
  let g = @moongrad.Graph::new()
  let p0 = g.new_param(2)
  let p1 = g.new_param(1)
  // The loss is (p0[0] - 1)^2 + (p0[1] - 2)^2 + (p1[0] - 3)^2.
  let e0 = g.sub(g.param(p0, 0), g.constant(1.0))
  let e1 = g.sub(g.param(p0, 1), g.constant(2.0))
  let e2 = g.sub(g.param(p1, 0), g.constant(3.0))
  let loss = g.add(g.add(g.mul(e0, e0), g.mul(e1, e1)), g.mul(e2, e2))
  let b = @moongrad.Binding::new()
  b.set_param(p0, [0.0, 0.0])
  b.set_param(p1, [0.0])
  let (_grads, program) = @moongrad.grad(g, b, loss)
  let flat = @moongrad.flat_param_gradients(program, b)
  // Three entries: two from the first slot and one from the second.
  assert_eq(flat.length(), 3)
  assert_close(flat[0], -2.0, 1.0e-12, "the first parameter's gradient")
  assert_close(flat[1], -4.0, 1.0e-12, "the second parameter's gradient")
  assert_close(flat[2], -6.0, 1.0e-12, "the third parameter's gradient")
}

///|
test "the_forward_dual_evaluation_carries_the_value_and_the_tangent" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let square = g.mul(x, x)
  let b = @moongrad.Binding::new()
  b.set_var(0, 3.0)
  // The direction is a variable slot: the tangent is one along it.
  let values = @moongrad.forward_dual(g, b, 0)
  assert_eq(values.length(), 2)
  assert_close(values[x].value, 3.0, 1.0e-12, "the variable's value")
  assert_close(values[x].tangent, 1.0, 1.0e-12, "the variable's tangent")
  // x^2 has the value 9 and the tangent 2x along the same direction.
  assert_close(values[square].value, 9.0, 1.0e-12, "the square's value")
  assert_close(values[square].tangent, 6.0, 1.0e-12, "the square's tangent")
}

///|
test "the_scalar_mirror_has_the_same_nodes_and_leaves_the_primal_alone" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let square = g.mul(x, x)
  let before = g.len()
  let h = @moongrad.mirror(g)
  // The mirror has the same nodes, at the same slots.
  assert_eq(h.len(), before)
  // And the primal is untouched: the mirror is a separate graph.
  assert_eq(g.len(), before)
  let b = @moongrad.Binding::new()
  b.set_var(0, 3.0)
  assert_close(
    @moongrad.forward(h, b, square).output,
    9.0,
    1.0e-12,
    "the mirror's value",
  )
}

///|
test "the_normalisation_centres_and_scales_along_the_axis" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let gamma = g.constant(@moongrad.Tensor::filled([2], 2.0))
  let beta = g.constant(@moongrad.Tensor::filled([2], 1.0))
  let out = @moongrad.normalize(g, x, 1, gamma, beta, 0.0)
  let b = @moongrad.TBinding::new()
  b.set_input(
    0,
    match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
      Some(t) => t
      None => return
    },
  )
  let values = @moongrad.t_forward(g, b)
  let y = values[out]
  // Each row has a mean of 1.5 or 3.5 and a standard deviation of
  // 0.5, so the normalised rows are [-1, 1] and [-1, 1]; the scale
  // is two and the shift is one.
  assert_close(y.data[0], -1.0, 1.0e-12, "row 0, feature 0")
  assert_close(y.data[1], 3.0, 1.0e-12, "row 0, feature 1")
  assert_close(y.data[2], -1.0, 1.0e-12, "row 1, feature 0")
  assert_close(y.data[3], 3.0, 1.0e-12, "row 1, feature 1")
}

///|
test "the_shape_mismatch_error_names_both_shapes" {
  let e = @moongrad.shape_mismatch("[2, 3]", "[3, 2]")
  let text = e.describe()
  check(
    text.contains("[2, 3]"),
    "the error should name the shape it was compiled for: \{text}",
  )
  check(
    text.contains("[3, 2]"),
    "the error should name the shape it was given: \{text}",
  )
}

///|
test "the_tensor_binary_operator_broadcasts_its_operands" {
  let x = @moongrad.Tensor::vector([1.0, 2.0, 3.0])
  let y = @moongrad.Tensor::vector([4.0, 5.0, 6.0])
  let sum = match @moongrad.t_binary(@moongrad.BinaryOp::Add, x, y) {
    Some(t) => t
    None => {
      check(false, "the addition should exist")
      return
    }
  }
  assert_close(sum.data[0], 5.0, 1.0e-12, "the sum")
  assert_close(sum.data[2], 9.0, 1.0e-12, "the sum")
  // A row against a vector keeps the broadcast shape.
  let row = @moongrad.Tensor::vector([10.0])
  let scaled = match @moongrad.t_binary(@moongrad.BinaryOp::Mul, x, row) {
    Some(t) => t
    None => {
      check(false, "the product should exist")
      return
    }
  }
  assert_eq(scaled.shape.length(), 1)
  assert_eq(scaled.data.length(), 3)
  assert_close(scaled.data[1], 20.0, 1.0e-12, "the broadcast product")
  // Shapes that do not broadcast have no answer.
  check(
    @moongrad.t_binary(
      @moongrad.BinaryOp::Add,
      @moongrad.Tensor::vector([1.0, 2.0]),
      @moongrad.Tensor::vector([1.0, 2.0, 3.0]),
    )
    is None,
    "two and three do not broadcast",
  )
}

///|
test "the_col2im_accumulates_the_overlapping_patches" {
  // One channel, a 3x3 input, a 2x2 kernel and a stride of one, so
  // there are four output positions and four columns a row.
  let grad = @moongrad.Tensor::filled([4, 4], 1.0)
  let out = @moongrad.t_col2im(grad, 1, 3, 3, 2, 2, 1, 2, 2, 1, 1)
  assert_eq(out.shape.length(), 4)
  assert_eq(out.data.length(), 9)
  // Every patch contributes one to each of its four elements, so a
  // corner is covered once, an edge twice and the middle four
  // times.
  assert_close(out.data[0], 1.0, 1.0e-12, "the top-left corner")
  assert_close(out.data[1], 2.0, 1.0e-12, "the top edge")
  assert_close(out.data[2], 1.0, 1.0e-12, "the top-right corner")
  assert_close(out.data[3], 2.0, 1.0e-12, "the left edge")
  assert_close(out.data[4], 4.0, 1.0e-12, "the middle")
  assert_close(out.data[5], 2.0, 1.0e-12, "the right edge")
  assert_close(out.data[6], 1.0, 1.0e-12, "the bottom-left corner")
  assert_close(out.data[7], 2.0, 1.0e-12, "the bottom edge")
  assert_close(out.data[8], 1.0, 1.0e-12, "the bottom-right corner")
}

///|
test "the_crop_takes_a_window_out_of_the_tensor" {
  let x = @moongrad.Tensor::vector([1.0, 2.0, 3.0, 4.0, 5.0])
  let cropped = match @moongrad.t_crop_to(x, [2], 0, 1) {
    Some(t) => t
    None => {
      check(false, "the crop should exist")
      return
    }
  }
  assert_eq(cropped.shape.length(), 1)
  assert_eq(cropped.data.length(), 2)
  assert_close(cropped.data[0], 2.0, 1.0e-12, "the first cropped value")
  assert_close(cropped.data[1], 3.0, 1.0e-12, "the second cropped value")
  // A window that runs off the end is refused.
  check(
    @moongrad.t_crop_to(x, [3], 0, 3) is None,
    "a window that runs off the end should be refused",
  )
  // And so is a target with a different number of axes.
  check(
    @moongrad.t_crop_to(x, [1, 2], 0, 0) is None,
    "a target of the wrong rank should be refused",
  )
}

///|
test "the_expanded_gradient_repeats_along_the_reduced_axis" {
  let gbar = match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
    Some(t) => t
    None => {
      check(false, "the reduced gradient should exist")
      return
    }
  }
  let out = @moongrad.t_expand_axis(gbar, 1, [2, 3, 2])
  assert_eq(out.shape.length(), 3)
  assert_eq(out.data.length(), 12)
  // Every element along the reduced axis maps to the same entry of
  // the reduced gradient.
  assert_close(out.data[0], 1.0, 1.0e-12, "the first element")
  assert_close(out.data[1], 2.0, 1.0e-12, "the second element")
  assert_close(out.data[2], 1.0, 1.0e-12, "the third element")
  assert_close(out.data[3], 2.0, 1.0e-12, "the fourth element")
  assert_close(out.data[6], 3.0, 1.0e-12, "the seventh element")
  assert_close(out.data[7], 4.0, 1.0e-12, "the eighth element")
  assert_close(out.data[10], 3.0, 1.0e-12, "the eleventh element")
}

///|
test "the_im2col_matrix_is_one_row_per_output_position" {
  let x = match
    @moongrad.Tensor::from_data([1, 1, 3, 3], [
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ]) {
    Some(t) => t
    None => {
      check(false, "the input should exist")
      return
    }
  }
  // One channel, a 2x2 kernel and a stride of one over a 3x3 input
  // gives four output positions and four columns a row.
  let cols = @moongrad.t_im2col(x, 2, 2, 1, 2, 2, 1, 1)
  assert_eq(cols.shape.length(), 2)
  assert_eq(cols.shape[0], 4)
  assert_eq(cols.shape[1], 4)
  // The first row is the patch at the top left.
  assert_close(cols.data[0], 1.0, 1.0e-12, "the patch's first element")
  assert_close(cols.data[1], 2.0, 1.0e-12, "the patch's second element")
  assert_close(cols.data[2], 4.0, 1.0e-12, "the patch's third element")
  assert_close(cols.data[3], 5.0, 1.0e-12, "the patch's fourth element")
  // The second row is the patch one column to the right.
  assert_close(cols.data[4], 2.0, 1.0e-12, "the next patch's first element")
  assert_close(cols.data[7], 6.0, 1.0e-12, "the next patch's fourth element")
}

///|
test "the_inverse_permutation_undoes_the_permutation" {
  let perm = [2, 0, 1]
  let inverse = @moongrad.t_inverse_perm(perm)
  assert_eq(inverse.length(), 3)
  assert_eq(inverse[0], 1)
  assert_eq(inverse[1], 2)
  assert_eq(inverse[2], 0)
  // Undoing it twice gives the original back.
  let back = @moongrad.t_inverse_perm(inverse)
  assert_eq(back[0], 2)
  assert_eq(back[1], 0)
  assert_eq(back[2], 1)
}

///|
test "the_dropout_mask_is_deterministic_and_depends_on_the_call_count" {
  let first = @moongrad.t_keep_mask(0.5, 7, 0, [20])
  let again = @moongrad.t_keep_mask(0.5, 7, 0, [20])
  assert_eq(first.data.length(), 20)
  for i = 0; i < first.data.length(); i = i + 1 {
    // The mask is made of zeroes and ones.
    check(
      first.data[i] == 0.0 || first.data[i] == 1.0,
      "the mask should be made of zeroes and ones",
    )
    // The same seed and the same call count give the same mask.
    check(first.data[i] == again.data[i], "the mask should be the same")
  }
  // A different call count gives a different mask.
  let other = @moongrad.t_keep_mask(0.5, 7, 1, [20])
  let mut differs = false
  for i = 0; i < first.data.length(); i = i + 1 {
    if first.data[i] != other.data[i] {
      differs = true
    }
  }
  check(differs, "a different call count should give a different mask")
}

///|
test "the_log_softmax_is_the_logarithm_of_the_softmax" {
  let x = @moongrad.Tensor::vector([1.0, 2.0, 3.0])
  let out = match @moongrad.t_log_softmax_axis(x, 0) {
    Some(t) => t
    None => {
      check(false, "the log softmax should exist")
      return
    }
  }
  let total = @math.exp(1.0) + @math.exp(2.0) + @math.exp(3.0)
  assert_close(out.data[0], 1.0 - @math.ln(total), 1.0e-12, "the first entry")
  assert_close(out.data[2], 3.0 - @math.ln(total), 1.0e-12, "the last entry")
  // The maximum is subtracted before the exponential, so a confident
  // and wrong prediction still gives a finite answer.
  let big = @moongrad.Tensor::vector([1000.0, 1001.0])
  let finite = match @moongrad.t_log_softmax_axis(big, 0) {
    Some(t) => t
    None => {
      check(false, "the log softmax should exist for large inputs")
      return
    }
  }
  check(finite.data[0].is_finite(), "the log softmax should stay finite")
  // An axis that is not there is refused.
  check(
    @moongrad.t_log_softmax_axis(x, 3) is None,
    "an axis out of range should be refused",
  )
}

///|
test "the_tensor_mirror_has_the_same_nodes_and_leaves_the_primal_alone" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let square = g.mul(x, x)
  let before = g.len()
  let h = @moongrad.t_mirror(g)
  // The mirror has the same nodes, at the same slots.
  assert_eq(h.len(), before)
  // And the primal is untouched: the mirror is a separate graph.
  assert_eq(g.len(), before)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([3.0]))
  let values = @moongrad.t_forward(h, b)
  assert_close(values[square].data[0], 9.0, 1.0e-12, "the mirror's value")
}

///|
test "the_elementwise_product_keeps_the_broadcast_shape" {
  let x = @moongrad.Tensor::vector([1.0, 2.0, 3.0])
  let y = @moongrad.Tensor::vector([4.0, 5.0, 6.0])
  let same = match @moongrad.t_mul_or(x, y) {
    Some(t) => t
    None => {
      check(false, "the product should exist")
      return
    }
  }
  assert_close(same.data[0], 4.0, 1.0e-12, "the product")
  assert_close(same.data[2], 18.0, 1.0e-12, "the product")
  // A matrix against a row keeps the matrix's shape.
  let m = match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
    Some(t) => t
    None => {
      check(false, "the matrix should exist")
      return
    }
  }
  let row = @moongrad.Tensor::vector([2.0, 3.0, 4.0])
  let broadcast = match @moongrad.t_mul_or(m, row) {
    Some(t) => t
    None => {
      check(false, "the broadcast product should exist")
      return
    }
  }
  assert_eq(broadcast.shape.length(), 2)
  assert_eq(broadcast.shape[0], 2)
  assert_eq(broadcast.shape[1], 3)
  assert_close(broadcast.data[0], 2.0, 1.0e-12, "the first element")
  assert_close(broadcast.data[4], 15.0, 1.0e-12, "the fifth element")
  // Shapes that do not broadcast have no product.
  check(
    @moongrad.t_mul_or(
      @moongrad.Tensor::vector([1.0, 2.0]),
      @moongrad.Tensor::vector([1.0, 2.0, 3.0]),
    )
    is None,
    "two and three do not broadcast",
  )
}

///|
test "the_pad_puts_the_tensor_inside_a_larger_one" {
  let x = @moongrad.Tensor::vector([1.0, 2.0])
  let padded = match @moongrad.t_pad_to(x, [4], 0, 1) {
    Some(t) => t
    None => {
      check(false, "the pad should exist")
      return
    }
  }
  assert_eq(padded.shape.length(), 1)
  assert_eq(padded.data.length(), 4)
  assert_close(padded.data[0], 0.0, 1.0e-12, "the space before")
  assert_close(padded.data[1], 1.0, 1.0e-12, "the first value")
  assert_close(padded.data[2], 2.0, 1.0e-12, "the second value")
  assert_close(padded.data[3], 0.0, 1.0e-12, "the space after")
  // A window that does not fit is refused.
  check(
    @moongrad.t_pad_to(x, [4], 0, 3) is None,
    "a window that runs off the end should be refused",
  )
}

///|
test "the_tensor_power_raises_every_element" {
  let x = @moongrad.Tensor::vector([2.0, 3.0, 4.0])
  let squares = @moongrad.t_pow(x, 2.0)
  assert_eq(squares.shape.length(), 1)
  assert_close(squares.data[0], 4.0, 1.0e-12, "the first square")
  assert_close(squares.data[2], 16.0, 1.0e-12, "the last square")
  // Zero to a negative power is not a number.
  let impossible = @moongrad.t_pow(@moongrad.Tensor::vector([0.0]), -1.0)
  check(
    !impossible.data[0].is_finite(),
    "zero to a negative power should not be a number",
  )
}

///|
test "the_tensor_scale_multiplies_every_element" {
  let x = @moongrad.Tensor::vector([1.0, -2.0, 3.0])
  let scaled = @moongrad.t_scale(x, 2.5)
  assert_eq(scaled.data.length(), 3)
  assert_close(scaled.data[0], 2.5, 1.0e-12, "the first element")
  assert_close(scaled.data[1], -5.0, 1.0e-12, "the second element")
  assert_close(scaled.data[2], 7.5, 1.0e-12, "the third element")
}

///|
test "the_scattered_row_lands_in_the_row_it_names" {
  let x = @moongrad.Tensor::vector([1.0, 2.0])
  let out = match @moongrad.t_scatter_row(x, [3, 2], 1, 2) {
    Some(t) => t
    None => {
      check(false, "the scatter should exist")
      return
    }
  }
  assert_eq(out.shape.length(), 2)
  assert_eq(out.data.length(), 6)
  assert_close(out.data[0], 0.0, 1.0e-12, "the empty row")
  assert_close(out.data[1], 0.0, 1.0e-12, "the empty row")
  assert_close(out.data[2], 1.0, 1.0e-12, "the scattered row")
  assert_close(out.data[3], 2.0, 1.0e-12, "the scattered row")
  assert_close(out.data[4], 0.0, 1.0e-12, "the empty row")
  assert_close(out.data[5], 0.0, 1.0e-12, "the empty row")
  // A row past the end is refused.
  check(
    @moongrad.t_scatter_row(x, [3, 2], 3, 2) is None,
    "a row past the end should be refused",
  )
  // And so is a row of the wrong width.
  check(
    @moongrad.t_scatter_row(x, [3, 2], 0, 3) is None,
    "a row of the wrong width should be refused",
  )
}

///|
test "the_sum_to_shape_undoes_the_broadcast" {
  let t = match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
    Some(x) => x
    None => {
      check(false, "the matrix should exist")
      return
    }
  }
  // A row of three against a 2x3 matrix was repeated along the first
  // axis, so the gradient is summed back down to a row.
  let row = match @moongrad.t_sum_to_shape(t, [3]) {
    Some(x) => x
    None => {
      check(false, "the sum should exist")
      return
    }
  }
  assert_eq(row.shape.length(), 1)
  assert_eq(row.data.length(), 3)
  assert_close(row.data[0], 5.0, 1.0e-12, "the first column")
  assert_close(row.data[2], 9.0, 1.0e-12, "the last column")
  // A target with a length-one axis keeps the axis.
  let kept = match @moongrad.t_sum_to_shape(t, [1, 3]) {
    Some(x) => x
    None => {
      check(false, "the sum should exist")
      return
    }
  }
  assert_eq(kept.shape.length(), 2)
  assert_eq(kept.shape[0], 1)
  assert_close(kept.data[2], 9.0, 1.0e-12, "the last column")
  // A target with more axes than the tensor cannot be reached.
  check(
    @moongrad.t_sum_to_shape(t, [1, 2, 3]) is None,
    "a sum cannot create an axis",
  )
  // And axes that do not line up are refused too.
  check(
    @moongrad.t_sum_to_shape(t, [4]) is None,
    "axes that do not line up should be refused",
  )
}

///|
test "the_tensor_unary_operator_preserves_the_shape" {
  let x = @moongrad.Tensor::vector([0.5, 1.5])
  let e = @moongrad.t_unary(@moongrad.UnaryOp::Exp, x)
  assert_eq(e.shape.length(), 1)
  assert_eq(e.data.length(), 2)
  assert_close(e.data[0], @math.exp(0.5), 1.0e-12, "the exponential")
  let r = @moongrad.t_unary(
    @moongrad.UnaryOp::Relu,
    @moongrad.Tensor::vector([-1.0, 2.0]),
  )
  assert_close(r.data[0], 0.0, 1.0e-12, "the relu of a negative")
  assert_close(r.data[1], 2.0, 1.0e-12, "the relu of a positive")
  // A logarithm of a number that is not positive is not a number.
  let l = @moongrad.t_unary(
    @moongrad.UnaryOp::Ln,
    @moongrad.Tensor::vector([-1.0]),
  )
  check(!l.data[0].is_finite(), "the log of a negative should not be a number")
}
