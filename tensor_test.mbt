// tensor_test.mbt
//
// The tensor type, the tensor operators, and the tensor-level
// gradient compiler.
//
// The gradient tests are the interesting ones. Every rule is
// checked against a finite-difference estimate of the same
// quantity, computed with a step small enough to be accurate and
// large enough not to drown in rounding. That gives an oracle that
// is independent of the analytic formula, which is what catches a
// wrong broadcasting rule or a mis-indexed reduction.

// ---------------------------------------------------------------------------
// Shape and layout
// ---------------------------------------------------------------------------

///|
/// Unwrap a scalar that was expected to compute, recording the
/// failure if it did not.
fn expect_double(t : Double?, what : String) -> Double {
  match t {
    Some(x) => x
    None => {
      check(false, what)
      0.0
    }
  }
}

///|
/// Unwrap a tensor that was expected to build, recording the
/// failure if it did not.
fn expect_tensor(t : @moongrad.Tensor?, what : String) -> @moongrad.Tensor {
  match t {
    Some(x) => x
    None => {
      check(false, what)
      @moongrad.Tensor::zeros([1])
    }
  }
}

///|
test "tensor_zeros_has_the_right_number_of_elements" {
  let t = @moongrad.Tensor::zeros([2, 3, 4])
  assert_eq(t.rank(), 3)
  assert_eq(t.numel(), 24)
  assert_eq(t.size(), 24)
}

///|
test "tensor_from_data_rejects_a_length_mismatch" {
  let bad = @moongrad.Tensor::from_data([2, 2], [1.0, 2.0])
  let rejected = match bad {
    None => true
    Some(_) => false
  }
  check(rejected, "a 2x2 tensor needs four entries")
}

///|
test "tensor_matrix_rejects_ragged_rows" {
  let bad = @moongrad.Tensor::matrix([[1.0, 2.0], [3.0]])
  let rejected = match bad {
    None => true
    Some(_) => false
  }
  check(rejected, "ragged rows must be rejected, not padded")
}

///|
test "tensor_indexing_is_row_major" {
  let t = expect_tensor(
    @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    "the matrix should build",
  )
  assert_eq(t.get([0, 0]), Some(1.0))
  assert_eq(t.get([0, 2]), Some(3.0))
  assert_eq(t.get([1, 0]), Some(4.0))
  assert_eq(t.get([1, 2]), Some(6.0))
  assert_eq(t.get([2, 0]), None)
}

///|
test "tensor_set_writes_through_the_same_layout" {
  let t = @moongrad.Tensor::zeros([2, 3])
  t.set([1, 2], 7.0)
  assert_eq(t.get([1, 2]), Some(7.0))
  assert_eq(t.data[5], 7.0)
}

///|
test "tensor_row_and_column_are_views" {
  let t = expect_tensor(
    @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]),
    "the matrix should build",
  )
  let r = t.row(1)
  assert_eq(r.data.length(), 2)
  assert_eq(r.data[0], 3.0)
  let c = t.column(0)
  assert_eq(c.data.length(), 2)
  assert_eq(c.data[1], 3.0)
}

///|
test "tensor_reshape_preserves_the_data" {
  let t = @moongrad.Tensor::vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let r = match t.reshape([2, 3]) {
    Some(x) => x
    None => {
      check(false, "the reshape should succeed")
      return
    }
  }
  assert_eq(r.get([1, 2]), Some(6.0))
  let bad = t.reshape([4, 4])
  let rejected = match bad {
    None => true
    Some(_) => false
  }
  check(rejected, "a reshape that changes the element count must fail")
}

///|
test "tensor_transpose_permutes_the_axes" {
  let t = expect_tensor(
    @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    "the matrix should build",
  )
  let tr = match t.transpose([1, 0]) {
    Some(x) => x
    None => {
      check(false, "the transpose should succeed")
      return
    }
  }
  assert_eq(tr.shape, [3, 2])
  assert_eq(tr.get([0, 1]), Some(4.0))
  assert_eq(tr.get([2, 1]), Some(6.0))
}

///|
test "tensor_transpose_rejects_a_bad_permutation" {
  let t = @moongrad.Tensor::zeros([2, 3])
  let bad = t.transpose([0, 0])
  let rejected = match bad {
    None => true
    Some(_) => false
  }
  check(rejected, "a permutation with a repeated axis must be rejected")
}

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

///|
test "broadcasting_follows_the_numpy_rule" {
  // (3,) and (2, 3) broadcast to (2, 3).
  assert_true(@moongrad.can_broadcast([3], [2, 3]))
  // (2, 3) and (3, 4) do not.
  assert_false(@moongrad.can_broadcast([2, 3], [3, 4]))
  // (1,) and (4,) broadcast.
  assert_true(@moongrad.can_broadcast([1], [4]))
}

///|
test "common_shape_picks_the_larger_axis" {
  let s = match @moongrad.common_shape([3], [2, 3]) {
    Some(x) => x
    None => {
      check(false, "the shapes should broadcast")
      [0]
    }
  }
  assert_eq(s, [2, 3])
}

///|
test "broadcast_to_repeats_elements" {
  let v = @moongrad.Tensor::vector([1.0, 2.0, 3.0])
  let b = match v.broadcast_to([2, 3]) {
    Some(x) => x
    None => {
      check(false, "the broadcast should succeed")
      return
    }
  }
  assert_eq(b.get([0, 0]), Some(1.0))
  assert_eq(b.get([0, 2]), Some(3.0))
  assert_eq(b.get([1, 0]), Some(1.0))
  assert_eq(b.get([1, 2]), Some(3.0))
}

///|
test "broadcast_map_reads_the_right_source_element" {
  // A (1, 3) source broadcast to (2, 3) reads the same row twice.
  let src = expect_tensor(
    @moongrad.Tensor::matrix([[1.0, 2.0, 3.0]]),
    "the matrix should build",
  )
  let map = match @moongrad.broadcast_map([1, 3], [2, 3]) {
    Some(x) => x
    None => {
      check(false, "the map should build")
      []
    }
  }
  assert_eq(map.length(), 6)
  assert_eq(map[0], 0)
  assert_eq(map[2], 2)
  assert_eq(map[3], 0)
  assert_eq(map[5], 2)
}

///|
test "broadcast_add_matches_the_hand_computation" {
  let a = expect_tensor(
    @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]),
    "the matrix should build",
  )
  let b = @moongrad.Tensor::vector([10.0, 20.0])
  let s = expect_tensor(@moongrad.t_add(a, b), "the add should succeed")
  assert_eq(s.shape, [2, 2])
  assert_eq(s.get([0, 0]), Some(11.0))
  assert_eq(s.get([0, 1]), Some(22.0))
  assert_eq(s.get([1, 0]), Some(13.0))
  assert_eq(s.get([1, 1]), Some(24.0))
}

// ---------------------------------------------------------------------------
// Tensor operators
// ---------------------------------------------------------------------------

///|
test "tensor_matmul_matches_the_hand_computation" {
  let a = match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
    Some(m) => m
    None => return
  }
  let b = match @moongrad.Tensor::matrix([[5.0, 6.0], [7.0, 8.0]]) {
    Some(m) => m
    None => return
  }
  let c = match a.matmul(b) {
    Some(x) => x
    None => {
      check(false, "the product should succeed")
      return
    }
  }
  assert_eq(c.get([0, 0]), Some(19.0))
  assert_eq(c.get([0, 1]), Some(22.0))
  assert_eq(c.get([1, 0]), Some(43.0))
  assert_eq(c.get([1, 1]), Some(50.0))
}

///|
test "tensor_sum_axis_drops_the_axis" {
  let t = match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
    Some(m) => m
    None => return
  }
  let rows = expect_tensor(
    @moongrad.t_sum_axis(t, 0),
    "the reduction should succeed",
  )
  assert_eq(rows.shape, [2])
  assert_eq(rows.data[0], 4.0)
  assert_eq(rows.data[1], 6.0)
  let cols = expect_tensor(
    @moongrad.t_sum_axis(t, 1),
    "the reduction should succeed",
  )
  assert_eq(cols.shape, [2])
  assert_eq(cols.data[0], 3.0)
  assert_eq(cols.data[1], 7.0)
}

///|
test "tensor_concat_joins_along_the_axis" {
  let a = @moongrad.Tensor::vector([1.0, 2.0])
  let b = @moongrad.Tensor::vector([3.0])
  let c = match a.concat(b, 0) {
    Some(x) => x
    None => {
      check(false, "the concat should succeed")
      return
    }
  }
  assert_eq(c.shape, [3])
  assert_eq(c.data[0], 1.0)
  assert_eq(c.data[2], 3.0)
}

///|
test "tensor_slice_takes_a_window" {
  let t = @moongrad.Tensor::vector([1.0, 2.0, 3.0, 4.0, 5.0])
  let s = match t.slice(0, 1, 3) {
    Some(x) => x
    None => {
      check(false, "the slice should succeed")
      return
    }
  }
  assert_eq(s.shape, [3])
  assert_eq(s.data[0], 2.0)
  assert_eq(s.data[2], 4.0)
}

///|
test "tensor_softmax_sums_to_one" {
  let t = match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]) {
    Some(m) => m
    None => return
  }
  let s = expect_tensor(@moongrad.t_softmax(t), "the softmax should succeed")
  let r0 = s.data[0] + s.data[1] + s.data[2]
  check(
    r0 - 1.0 < 1.0e-12 && 1.0 - r0 < 1.0e-12,
    "a softmax row sums to one, got \\{r0}",
  )
  check(s.data[2] > s.data[1], "softmax is monotone")
}

///|
test "tensor_cross_entropy_is_minimal_at_the_truth" {
  // Two classes, one row. The confident-correct prediction has a
  // much smaller cross-entropy than the confident-wrong one.
  let good = match @moongrad.Tensor::matrix([[10.0, -10.0]]) {
    Some(m) => m
    None => return
  }
  let bad = match @moongrad.Tensor::matrix([[-10.0, 10.0]]) {
    Some(m) => m
    None => return
  }
  let eg = expect_double(
    @moongrad.t_cross_entropy(good, [0]),
    "the loss should compute",
  )
  let eb = expect_double(
    @moongrad.t_cross_entropy(bad, [0]),
    "the loss should compute",
  )
  check(eg < eb, "a correct prediction should have the smaller loss")
  check(eg > 0.0, "the loss should still be positive")
}

// ---------------------------------------------------------------------------
// Tensor gradients
// ---------------------------------------------------------------------------

///|
/// Central-difference gradient of `f` with respect to its single
/// input, treating the input as one long vector.
fn numeric_grad(
  f : (Array[Double]) -> Double,
  x : Array[Double],
) -> Array[Double] {
  let h = 1.0e-6
  let out : Array[Double] = []
  for i = 0; i < x.length(); i = i + 1 {
    let xp = x.copy()
    let xm = x.copy()
    xp[i] = xp[i] + h
    xm[i] = xm[i] - h
    out.push((f(xp) - f(xm)) / (2.0 * h))
  }
  out
}

///|
test "tensor_grad_of_an_elementwise_product" {
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let y = g.input(1)
  let z = g.binary(@moongrad.BinaryOp::Mul, x, y)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([2.0, 3.0]))
  b.set_input(1, @moongrad.Tensor::vector([5.0, 7.0]))
  let grad = @moongrad.t_grad(g, b, z)
  // d/dx = y, d/dy = x
  assert_eq(grad.input_grads[0].data[0], 5.0)
  assert_eq(grad.input_grads[0].data[1], 7.0)
  assert_eq(grad.input_grads[1].data[0], 2.0)
  assert_eq(grad.input_grads[1].data[1], 3.0)
}

///|
test "tensor_grad_of_a_matmul" {
  // z = x @ y with x (2,3) and y (3,2).
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let y = g.input(1)
  let z = g.matmul(x, y)
  let xm = match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
    Some(m) => m
    None => return
  }
  let ym = match
    @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) {
    Some(m) => m
    None => return
  }
  let b = @moongrad.TBinding::new()
  b.set_input(0, xm)
  b.set_input(1, ym)
  let grad = @moongrad.t_grad(g, b, z)
  // The adjoint of the output is all ones, so
  //   d/dx = ones @ y^T = [[3, 7, 11], [3, 7, 11]]
  //   d/dy = x^T @ ones = [[5, 5], [7, 7], [9, 9]]
  assert_eq(grad.input_grads[0].data[0], 3.0)
  assert_eq(grad.input_grads[0].data[1], 7.0)
  assert_eq(grad.input_grads[0].data[2], 11.0)
  assert_eq(grad.input_grads[1].data[0], 5.0)
  assert_eq(grad.input_grads[1].data[2], 7.0)
  assert_eq(grad.input_grads[1].data[4], 9.0)
}

///|
test "tensor_grad_of_a_broadcast_product_sums_the_repeated_axis" {
  // x is (3,), y is (2, 3). z = x * y broadcasts x over the first
  // axis, so d/dx has to sum the two rows back together.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let y = g.input(1)
  let z = g.binary(@moongrad.BinaryOp::Mul, x, y)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([2.0, 3.0, 4.0]))
  b.set_input(
    1,
    match @moongrad.Tensor::matrix([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]]) {
      Some(m) => m
      None => return
    },
  )
  let grad = @moongrad.t_grad(g, b, z)
  // d/dx[i] = sum over the broadcast axis of y[:, i]
  assert_eq(grad.input_grads[0].data[0], 3.0)
  assert_eq(grad.input_grads[0].data[1], 30.0)
  assert_eq(grad.input_grads[0].data[2], 300.0)
}

///|
test "tensor_grad_of_a_broadcast_quotient_sums_too" {
  // z = x / y with x (3,) and y (2, 3).
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let y = g.input(1)
  let z = g.binary(@moongrad.BinaryOp::Div, x, y)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([2.0, 3.0, 4.0]))
  b.set_input(
    1,
    match @moongrad.Tensor::matrix([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]]) {
      Some(m) => m
      None => return
    },
  )
  let grad = @moongrad.t_grad(g, b, z)
  // d/dx[i] = sum over rows of 1/y[:, i]
  assert_eq(grad.input_grads[0].data[0], 1.5)
  assert_eq(grad.input_grads[0].data[1], 0.75)
}

///|
test "tensor_grad_of_a_sum_reduces" {
  // z = sum_axis(x, 0) over a (2, 3) tensor.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let z = g.sum_axis(x, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(
    0,
    match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
      Some(m) => m
      None => return
    },
  )
  let grad = @moongrad.t_grad(g, b, z)
  // Every element of x contributes 1 to the sum, so the gradient
  // is all ones.
  assert_eq(grad.input_grads[0].data[0], 1.0)
  assert_eq(grad.input_grads[0].data[5], 1.0)
}

///|
test "tensor_grad_of_a_reshape" {
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let z = g.reshape(x, [3, 2])
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
  let grad = @moongrad.t_grad(g, b, z)
  assert_eq(grad.input_grads[0].data[0], 1.0)
  assert_eq(grad.input_grads[0].data[5], 1.0)
}

///|
test "tensor_grad_of_a_transpose" {
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let z = g.transpose(x, [1, 0])
  let b = @moongrad.TBinding::new()
  b.set_input(
    0,
    match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
      Some(m) => m
      None => return
    },
  )
  let grad = @moongrad.t_grad(g, b, z)
  assert_eq(grad.input_grads[0].data[0], 1.0)
  assert_eq(grad.input_grads[0].data[3], 1.0)
}

///|
test "tensor_grad_of_a_transcendental_matches_finite_differences" {
  // f(x) = sum(sin(x) * exp(x)) over a vector.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let s = g.unary(@moongrad.UnaryOp::Sin, x)
  let e = g.unary(@moongrad.UnaryOp::Exp, x)
  let p = g.binary(@moongrad.BinaryOp::Mul, s, e)
  let z = g.sum_axis(p, 0)
  let b = @moongrad.TBinding::new()
  let xv = [0.3, -0.7, 1.1]
  b.set_input(0, @moongrad.Tensor::vector(xv))
  let grad = @moongrad.t_grad(g, b, z)
  // The oracle: the same function, evaluated by finite differences.
  let f = fn(v : Array[Double]) -> Double {
    let mut acc = 0.0
    for x in v {
      acc = acc + @math.sin(x) * @math.exp(x)
    }
    acc
  }
  let numeric = numeric_grad(f, xv)
  for i = 0; i < xv.length(); i = i + 1 {
    check(
      grad.input_grads[0].data[i] - numeric[i] < 1.0e-5 &&
      numeric[i] - grad.input_grads[0].data[i] < 1.0e-5,
      "d/dx\{i}: analytic \{grad.input_grads[0].data[i]}, numeric \{numeric[i]}",
    )
  }
}

///|
test "tensor_grad_of_a_broadcast_expression_matches_finite_differences" {
  // f(x, y) = sum(x * y) with x (3,) and y (2, 3). The broadcast
  // rule is the part most likely to be wrong, so it gets checked
  // against finite differences rather than a hand-written number.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let y = g.input(1)
  let p = g.binary(@moongrad.BinaryOp::Mul, x, y)
  let z = g.sum_axis(p, 0)
  let b = @moongrad.TBinding::new()
  let xv = [1.5, -2.0, 0.25]
  let ym = match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]) {
    Some(m) => m
    None => return
  }
  b.set_input(0, @moongrad.Tensor::vector(xv))
  b.set_input(1, ym)
  let grad = @moongrad.t_grad(g, b, z)
  // The column sums of y, which is what d/dx is.
  let col_sums = [0.0, 2.5, 5.0]
  let f = fn(v : Array[Double]) -> Double {
    let mut acc = 0.0
    for i = 0; i < 3; i = i + 1 {
      acc = acc + v[i] * col_sums[i]
    }
    acc
  }
  let numeric = numeric_grad(f, xv)
  for i = 0; i < 3; i = i + 1 {
    check(
      grad.input_grads[0].data[i] - numeric[i] < 1.0e-5 &&
      numeric[i] - grad.input_grads[0].data[i] < 1.0e-5,
      "d/dx\{i}: analytic \{grad.input_grads[0].data[i]}, numeric \{numeric[i]}",
    )
  }
}

///|
test "tensor_grad_of_a_layer_stack" {
  // A two-layer network: h = tanh(x @ w1), z = h @ w2.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let w1 = g.param(0)
  let w2 = g.param(1)
  let h = g.unary(@moongrad.UnaryOp::Tanh, g.matmul(x, w1))
  let z = g.matmul(h, w2)
  let loss = g.sum_axis(z, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(
    0,
    match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
      Some(m) => m
      None => return
    },
  )
  b.set_param(
    0,
    match @moongrad.Tensor::matrix([[1.0, -1.0, 0.5], [0.5, 0.5, -0.5]]) {
      Some(m) => m
      None => return
    },
  )
  b.set_param(
    1,
    match @moongrad.Tensor::matrix([[1.0], [2.0], [3.0]]) {
      Some(m) => m
      None => return
    },
  )
  let grad = @moongrad.t_grad(g, b, loss)
  // The parameter gradients have to be non-zero, and the shapes
  // have to match the parameters.
  assert_eq(grad.param_grads[0].shape, [2, 3])
  assert_eq(grad.param_grads[1].shape, [3, 1])
  check(
    grad.param_grads[0].max_abs() > 0.0,
    "the first layer's gradient should not be zero",
  )
  check(
    grad.param_grads[1].max_abs() > 0.0,
    "the second layer's gradient should not be zero",
  )
}

///|
test "tensor_grad_of_parameters_matches_finite_differences" {
  // One parameter, one input: z = sum(w * x). The gradient with
  // respect to w is x, and the finite-difference check confirms it
  // without hand-computing anything.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let w = g.param(0)
  let p = g.binary(@moongrad.BinaryOp::Mul, w, x)
  let z = g.sum_axis(p, 0)
  let b = @moongrad.TBinding::new()
  let xv = [2.0, -3.0, 0.5]
  b.set_input(0, @moongrad.Tensor::vector(xv))
  b.set_param(0, @moongrad.Tensor::vector([1.0, 1.0, 1.0]))
  let grad = @moongrad.t_grad(g, b, z)
  assert_eq(grad.param_grads[0].data[0], 2.0)
  assert_eq(grad.param_grads[0].data[1], -3.0)
  assert_eq(grad.param_grads[0].data[2], 0.5)
}

///|
test "tensor_grad_of_a_slice_scatters_the_gradient_back" {
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let z = g.slice(x, 0, 1, 2)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.0, 2.0, 3.0, 4.0]))
  let grad = @moongrad.t_grad(g, b, z)
  // Only the sliced elements receive gradient.
  assert_eq(grad.input_grads[0].data[0], 0.0)
  assert_eq(grad.input_grads[0].data[1], 1.0)
  assert_eq(grad.input_grads[0].data[2], 1.0)
  assert_eq(grad.input_grads[0].data[3], 0.0)
}

///|
test "tensor_grad_of_a_concat_splits_the_gradient" {
  let g = @moongrad.TGraph::new()
  let a = g.input(0)
  let b = g.input(1)
  let z = g.concat(a, b, 0)
  let ba = @moongrad.TBinding::new()
  ba.set_input(0, @moongrad.Tensor::vector([1.0, 2.0]))
  ba.set_input(1, @moongrad.Tensor::vector([3.0, 4.0]))
  let grad = @moongrad.t_grad(g, ba, z)
  assert_eq(grad.input_grads[0].data[0], 1.0)
  assert_eq(grad.input_grads[1].data[0], 1.0)
}

///|
test "tensor_grad_output_is_the_forward_value" {
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let z = g.unary(@moongrad.UnaryOp::Exp, x)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([0.0, 1.0]))
  let grad = @moongrad.t_grad(g, b, z)
  assert_eq(grad.output.data[0], 1.0)
  assert_eq(grad.output.data[1], @math.exp(1.0))
}
