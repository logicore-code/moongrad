// softmax_test.mbt
//
// The tensor-level softmax.
//
// The softmax is the first operation in the tensor engine whose
// derivative is not elementwise: the Jacobian of a row couples every
// entry of that row with every other. So the tests below check the
// value against the plain formula, the gradient against a finite
// difference of the forward pass, and — the one that matters most —
// that a gradient compiled at one input is still correct at another.
//
// A softmax that froze its own value into the gradient program
// would pass the first test and fail the last one.

///|
/// A binding with a 2x3 matrix in slot 0.
fn softmax_binding(vals : Array[Double]) -> @moongrad.TBinding {
  let b = @moongrad.TBinding::new()
  let t = match @moongrad.Tensor::from_data([2, 3], vals) {
    Some(x) => x
    None => @moongrad.Tensor::zeros([2, 3])
  }
  b.set_input(0, t)
  b
}

///|
/// The plain softmax of a row, computed independently of the
/// implementation under test.
fn plain_softmax(row : Array[Double]) -> Array[Double] {
  let mut m = row[0]
  for i = 1; i < row.length(); i = i + 1 {
    if row[i] > m {
      m = row[i]
    }
  }
  let mut sum = 0.0
  for i = 0; i < row.length(); i = i + 1 {
    sum = sum + @math.exp(row[i] - m)
  }
  let out : Array[Double] = []
  for i = 0; i < row.length(); i = i + 1 {
    out.push(@math.exp(row[i] - m) / sum)
  }
  out
}

///|
/// The graph `loss(x) = sum_ij softmax(x)[i,j] * x[i,j]`.
///
/// The product with the input keeps the gradient from degenerating:
/// the sum of a softmax row is one, so a loss that only summed the
/// softmax would have a zero gradient everywhere.
fn softmax_loss_graph() -> (@moongrad.TGraph, Int32) {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let s = g.softmax(x, 1)
  let prod = g.mul(s, x)
  let per_row = g.sum_axis(prod, 1)
  let total = g.sum_axis(per_row, 0)
  (g, total)
}

///|
test "the_softmax_matches_the_plain_formula" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let s = g.softmax(x, 1)
  let b = softmax_binding([1.0, 2.0, 3.0, -1.0, 0.5, 1.5])
  let values = @moongrad.t_forward(g, b)
  let out = values[s]
  let first = plain_softmax([1.0, 2.0, 3.0])
  let second = plain_softmax([-1.0, 0.5, 1.5])
  for i = 0; i < 3; i = i + 1 {
    assert_close(out.data[i], first[i], 1.0e-12, "the first row of the softmax")
    assert_close(
      out.data[3 + i],
      second[i],
      1.0e-12,
      "the second row of the softmax",
    )
  }
}

///|
test "a_softmax_row_sums_to_one" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let s = g.softmax(x, 1)
  let b = softmax_binding([1.0, 2.0, 3.0, -1.0, 0.5, 1.5])
  let values = @moongrad.t_forward(g, b)
  let out = values[s]
  let mut first = 0.0
  let mut second = 0.0
  for i = 0; i < 3; i = i + 1 {
    first = first + out.data[i]
    second = second + out.data[3 + i]
  }
  assert_close(first, 1.0, 1.0e-12, "the first row should sum to one")
  assert_close(second, 1.0, 1.0e-12, "the second row should sum to one")
}

///|
test "a_softmax_of_a_large_logit_does_not_overflow" {
  // exp(1000) is not a number, so the maximum has to be subtracted
  // before the exponential. A softmax that did not would return NaN
  // here, and the training run that contains it would fail in a way
  // that is very hard to trace back.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let s = g.softmax(x, 1)
  let b = softmax_binding([1000.0, 1001.0, 999.0, 0.0, 0.0, 0.0])
  let values = @moongrad.t_forward(g, b)
  let out = values[s]
  for i = 0; i < 6; i = i + 1 {
    check(
      out.data[i] == out.data[i],
      "a softmax entry should be a number, not a NaN",
    )
  }
  assert_close(out.data[0], 0.24472847, 1.0e-6, "the first entry")
  assert_close(out.data[1], 0.66524096, 1.0e-6, "the second entry")
}

///|
test "the_softmax_gradient_matches_a_finite_difference" {
  let (g, target) = softmax_loss_graph()
  let vals = [1.0, 2.0, 3.0, -1.0, 0.5, 1.5]
  let grads = @moongrad.t_grad(g, softmax_binding(vals), target)
  let h = 1.0e-6
  for i = 0; i < 6; i = i + 1 {
    let up = vals.copy()
    let down = vals.copy()
    up[i] = up[i] + h
    down[i] = down[i] - h
    let vu = @moongrad.t_forward(g, softmax_binding(up))
    let vd = @moongrad.t_forward(g, softmax_binding(down))
    let numeric = (vu[target].data[0] - vd[target].data[0]) / (2.0 * h)
    assert_close(
      grads.input_grads[0].data[i],
      numeric,
      1.0e-5,
      "the softmax gradient",
    )
  }
}

///|
test "the_softmax_gradient_matches_the_analytic_jacobian" {
  // The Jacobian of a softmax row is `s * (g - sum(g*s))`. The test
  // uses g = 1 on one row, whose gradient is known in closed form.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let s = g.softmax(x, 1)
  let b = softmax_binding([1.0, 2.0, 3.0, -1.0, 0.5, 1.5])
  let grads = @moongrad.t_grad(g, b, s)
  let values = @moongrad.t_forward(g, b)
  let out = values[s]
  // With an incoming gradient of one on the first row, the gradient
  // is s * (1 - sum(s)) = 0 for a row that sums to one, and on the
  // second row it is s * (1 - 1) = 0 as well. What is not zero is the
  // row sum of the incoming gradient, which is three on the first
  // row: the gradient of the sum of a softmax row is zero.
  let mut row = 0.0
  for i = 0; i < 3; i = i + 1 {
    row = row + out.data[i]
  }
  for i = 0; i < 3; i = i + 1 {
    assert_close(
      grads.input_grads[0].data[i],
      out.data[i] * (1.0 - row),
      1.0e-12,
      "the softmax jacobian",
    )
  }
}

///|
test "a_compiled_softmax_gradient_is_reusable_at_another_input" {
  // Compile at one input and evaluate at another. If the gradient
  // program had frozen the softmax's value at the compile point,
  // this would report the derivative of the wrong function.
  let (g, target) = softmax_loss_graph()
  let compiled = @moongrad.compile_tensor_gradient(
    g,
    softmax_binding([1.0, 2.0, 3.0, -1.0, 0.5, 1.5]),
    target,
  )
  let other = [-3.0, 0.0, 4.0, 2.0, -2.0, 1.0]
  let (grads, _) = @moongrad.evaluate_tensor_gradient(
    compiled,
    softmax_binding(other),
  )
  // Compare against a finite difference at the same point.
  let h = 1.0e-6
  for i = 0; i < 6; i = i + 1 {
    let up = other.copy()
    let down = other.copy()
    up[i] = up[i] + h
    down[i] = down[i] - h
    let vu = @moongrad.t_forward(g, softmax_binding(up))
    let vd = @moongrad.t_forward(g, softmax_binding(down))
    let numeric = (vu[target].data[0] - vd[target].data[0]) / (2.0 * h)
    assert_close(
      grads[0].data[i],
      numeric,
      1.0e-5,
      "the reused softmax gradient",
    )
  }
}

///|
test "a_softmax_of_an_empty_axis_is_refused" {
  let t = match @moongrad.Tensor::from_data([2, 0], []) {
    Some(x) => x
    None => return
  }
  let s = @moongrad.t_softmax_axis(t, 1)
  check(s is None, "a softmax over an empty axis should be refused")
}

///|
test "a_softmax_over_an_axis_that_does_not_exist_is_refused" {
  let t = @moongrad.Tensor::vector([1.0, 2.0])
  let s = @moongrad.t_softmax_axis(t, 5)
  check(s is None, "a softmax over a missing axis should be refused")
}
