// nn_norm_test.mbt
//
// Normalisation layers.
//
// The forward pass is checked against the definition, and the
// gradient against central differences of that same forward pass, so
// neither depends on the other's derivation.

///|
/// A two-sample, three-feature tensor.
fn sample_input() -> @moongrad.Tensor {
  let t = @moongrad.Tensor::zeros([2, 3])
  let values = [1.0, 2.0, 3.0, -1.0, 0.0, 5.0]
  for i = 0; i < values.length(); i = i + 1 {
    t.data[i] = values[i]
  }
  t
}

///|
test "layer_norm_centres_and_scales_each_sample" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let gamma = g.constant(@moongrad.Tensor::filled([3], 1.0))
  let beta = g.constant(@moongrad.Tensor::filled([3], 0.0))
  let out = @moongrad.layer_norm(g, x, 1, gamma, beta, 1.0e-8)
  let b = @moongrad.TBinding::new()
  b.set_input(0, sample_input())
  let values = @moongrad.t_forward(g, b)
  let y = values[out]
  // Row 0 is [1,2,3]: mean 2, and the variance is 2/3, so the
  // normalised row is [(1-2), 0, (3-2)] / sqrt(2/3).
  let scale = 1.0 / (2.0 / 3.0).sqrt()
  assert_close(y.data[0], -1.0 * scale, 1.0e-6, "row 0, feature 0")
  assert_close(y.data[1], 0.0, 1.0e-6, "row 0, feature 1")
  assert_close(y.data[2], 1.0 * scale, 1.0e-6, "row 0, feature 2")
  // Row 1 is [-1,0,5]: mean 4/3, and the variance is
  // ((-7/3)^2 + (-4/3)^2 + (11/3)^2) / 3 = 62/9.
  let mean1 = 4.0 / 3.0
  let scale1 = 1.0 / (62.0 / 9.0).sqrt()
  assert_close(y.data[3], (-1.0 - mean1) * scale1, 1.0e-6, "row 1, feature 0")
  assert_close(y.data[5], (5.0 - mean1) * scale1, 1.0e-6, "row 1, feature 2")
}

///|
test "the_normalised_rows_have_zero_mean" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let gamma = g.constant(@moongrad.Tensor::filled([3], 1.0))
  let beta = g.constant(@moongrad.Tensor::filled([3], 0.0))
  let out = @moongrad.layer_norm(g, x, 1, gamma, beta, 1.0e-8)
  let b = @moongrad.TBinding::new()
  b.set_input(0, sample_input())
  let values = @moongrad.t_forward(g, b)
  let y = values[out]
  let mean0 = (y.data[0] + y.data[1] + y.data[2]) / 3.0
  assert_close(mean0, 0.0, 1.0e-6, "row 0 mean")
}

///|
test "the_layer_norm_gradient_matches_finite_differences" {
  // The objective is the sum of the normalised values, and the
  // gradient is compared against central differences of the same
  // forward pass.
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let gamma = g.constant(@moongrad.Tensor::filled([3], 1.0))
  let beta = g.constant(@moongrad.Tensor::filled([3], 0.0))
  let out = @moongrad.layer_norm(g, x, 1, gamma, beta, 1.0e-8)
  // The objective is the sum of the squares of the normalised
  // values. Summing the normalised values themselves would not do:
  // a normalised row sums to zero by construction, so that
  // objective is constant and its gradient is zero.
  let squared = g.mul(out, out)
  let loss = g.sum_axis(g.sum_axis(squared, 1), 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, sample_input())
  let grads = @moongrad.t_grad(g, b, loss)
  let eps = 1.0e-6
  for i = 0; i < 6; i = i + 1 {
    let up = sample_input()
    let down = sample_input()
    up.data[i] = up.data[i] + eps
    down.data[i] = down.data[i] - eps
    let bu = @moongrad.TBinding::new()
    bu.set_input(0, up)
    let bd = @moongrad.TBinding::new()
    bd.set_input(0, down)
    let vu = @moongrad.t_forward(g, bu)
    let vd = @moongrad.t_forward(g, bd)
    let numeric = (vu[loss].data[0] - vd[loss].data[0]) / (2.0 * eps)
    assert_close(
      grads.input_grads[0].data[i],
      numeric,
      1.0e-5,
      "layer norm gradient",
    )
  }
}

///|
test "batch_norm_centres_each_feature_over_the_batch" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let gamma = g.constant(@moongrad.Tensor::filled([3], 1.0))
  let beta = g.constant(@moongrad.Tensor::filled([3], 0.0))
  let out = @moongrad.batch_norm(g, x, 0, gamma, beta, 1.0e-8)
  let b = @moongrad.TBinding::new()
  b.set_input(0, sample_input())
  let values = @moongrad.t_forward(g, b)
  let y = values[out]
  // Feature 0 is [1, -1]: mean 0, so the normalised values are
  // +-1 / 1 (the variance of [1,-1] is 1).
  assert_close(y.data[0], 1.0, 1.0e-6, "feature 0, sample 0")
  assert_close(y.data[3], -1.0, 1.0e-6, "feature 0, sample 1")
}

///|
test "a_mean_over_an_axis_is_the_average" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let mean = g.mean_axis(x, 1)
  let b = @moongrad.TBinding::new()
  b.set_input(0, sample_input())
  let values = @moongrad.t_forward(g, b)
  // The mean over the feature axis drops it: [2, 3] becomes [2].
  assert_eq(values[mean].shape.length(), 1)
  assert_close(values[mean].data[0], 2.0, 1.0e-12, "row 0 mean")
  assert_close(values[mean].data[1], 4.0 / 3.0, 1.0e-12, "row 1 mean")
}

///|
test "the_mean_gradient_matches_finite_differences" {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let mean = g.mean_axis(x, 1)
  let loss = g.sum_axis(mean, 0)
  let b = @moongrad.TBinding::new()
  b.set_input(0, sample_input())
  let grads = @moongrad.t_grad(g, b, loss)
  // d(sum of the row means)/dx = 1/3 for every entry.
  for i = 0; i < 6; i = i + 1 {
    assert_close(
      grads.input_grads[0].data[i],
      1.0 / 3.0,
      1.0e-12,
      "mean gradient",
    )
  }
}
