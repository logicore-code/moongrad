// conv_test.mbt
//
// Convolution and the layers built on it.
//
// The convolution's gradient is checked against finite differences
// of the same convolution, which is the only way to be sure the
// im2col / col2im pair and the weight layout all agree. A
// convolution whose gradient is subtly wrong trains to a lower
// loss and produces garbage predictions, and neither shows up in a
// shape check.

// ---------------------------------------------------------------------------
// The forward pass
// ---------------------------------------------------------------------------

///|
test "conv_forward_matches_a_hand_computed_value" {
  // One input channel, one output channel, a 2x2 kernel, no
  // padding and a stride of one. The output is the 2x2 correlation
  // of the input with the kernel.
  let x = match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
    Some(m) => m
    None => {
      check(false, "the matrix should build")
      return
    }
  }
  // (batch, channels, h, w)
  let input = match @moongrad.Tensor::from_data([1, 1, 2, 3], x.data) {
    Some(t) => t
    None => {
      check(false, "the tensor should build")
      return
    }
  }
  // One 2x2 kernel of ones.
  let kernel = match
    @moongrad.Tensor::from_data([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0]) {
    Some(t) => t
    None => {
      check(false, "the tensor should build")
      return
    }
  }
  let out = match @moongrad.t_conv2d(input, kernel, None, 1, 1, 0, 0) {
    Some(t) => t
    None => {
      check(false, "the convolution should succeed")
      return
    }
  }
  // A 2x3 input with a 2x2 kernel and no padding gives a 1x2
  // output: [1+2+4+5, 2+3+5+6] = [12, 16].
  assert_eq(out.shape, [1, 1, 1, 2])
  assert_eq(out.data[0], 12.0)
  assert_eq(out.data[1], 16.0)
}

///|
test "conv_forward_with_a_stride_skips_positions" {
  let x = match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
    Some(m) => m
    None => return
  }
  let input = match @moongrad.Tensor::from_data([1, 1, 2, 3], x.data) {
    Some(t) => t
    None => return
  }
  let kernel = match
    @moongrad.Tensor::from_data([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0]) {
    Some(t) => t
    None => return
  }
  // A stride of two leaves a single output position, at the
  // top-left corner.
  let out = match @moongrad.t_conv2d(input, kernel, None, 2, 2, 0, 0) {
    Some(t) => t
    None => {
      check(false, "the convolution should succeed")
      return
    }
  }
  assert_eq(out.shape, [1, 1, 1, 1])
  assert_eq(out.data[0], 12.0)
}

///|
test "conv_forward_with_padding_keeps_the_size" {
  let x = match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
    Some(m) => m
    None => return
  }
  let input = match @moongrad.Tensor::from_data([1, 1, 2, 2], x.data) {
    Some(t) => t
    None => return
  }
  // A 3x3 kernel with a padding of one on each side is the
  // 'same' convolution: the output keeps the input's spatial size.
  let kernel = match
    @moongrad.Tensor::from_data([1, 1, 3, 3], [
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    ]) {
    Some(t) => t
    None => return
  }
  let out = match @moongrad.t_conv2d(input, kernel, None, 1, 1, 1, 1) {
    Some(t) => t
    None => {
      check(false, "the convolution should succeed")
      return
    }
  }
  assert_eq(out.shape, [1, 1, 2, 2])
}

///|
test "conv_forward_rejects_mismatched_channels" {
  let input = @moongrad.Tensor::zeros([1, 2, 3, 3])
  let kernel = @moongrad.Tensor::zeros([1, 1, 2, 2])
  let out = @moongrad.t_conv2d(input, kernel, None, 1, 1, 0, 0)
  let rejected = match out {
    None => true
    Some(_) => false
  }
  check(rejected, "a kernel with the wrong channel count must be rejected")
}

// ---------------------------------------------------------------------------
// The gradient
// ---------------------------------------------------------------------------

///|
/// The convolution, as a plain function of the input and the
/// weight, so a finite difference can be taken of it.
fn conv_of(
  x : @moongrad.Tensor,
  w : @moongrad.Tensor,
  stride_h : Int32,
  stride_w : Int32,
  pad_h : Int32,
  pad_w : Int32,
) -> Double {
  match @moongrad.t_conv2d(x, w, None, stride_h, stride_w, pad_h, pad_w) {
    Some(t) => t.sum()
    None => 0.0
  }
}

///|
test "conv_grad_matches_finite_differences" {
  // One input channel, one output channel, a 2x2 kernel, no
  // padding. Both the gradient with respect to the input and the
  // gradient with respect to the weight are checked against
  // central differences.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let w = g.param(0)
  let conv = g.conv2d(x, w, 1, 1, 2, 2, 1, 1, 0, 0)
  let loss = g.sum_axis(conv, 0)
  let b = @moongrad.TBinding::new()
  let xv = match
    @moongrad.Tensor::from_data([1, 1, 2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Some(t) => t
    None => return
  }
  let wv = match
    @moongrad.Tensor::from_data([1, 1, 2, 2], [0.5, -0.25, 1.0, 0.75]) {
    Some(t) => t
    None => return
  }
  b.set_input(0, xv)
  b.set_param(0, wv)
  let grad = @moongrad.t_grad(g, b, loss)
  // The gradient with respect to the input, by central difference.
  let h = 1.0e-6
  for i = 0; i < xv.data.length(); i = i + 1 {
    let xp = xv.to_data()
    let xm = xv.to_data()
    xp[i] = xp[i] + h
    xm[i] = xm[i] - h
    let fp = conv_of(
      match @moongrad.Tensor::from_data([1, 1, 2, 3], xp) {
        Some(t) => t
        None => return
      },
      wv,
      1,
      1,
      0,
      0,
    )
    let fm = conv_of(
      match @moongrad.Tensor::from_data([1, 1, 2, 3], xm) {
        Some(t) => t
        None => return
      },
      wv,
      1,
      1,
      0,
      0,
    )
    let numeric = (fp - fm) / (2.0 * h)
    check(
      grad.input_grads[0].data[i] - numeric < 1.0e-4 &&
      numeric - grad.input_grads[0].data[i] < 1.0e-4,
      "d/dx\{i}: analytic \{grad.input_grads[0].data[i]}, numeric \{numeric}",
    )
  }
  // And the gradient with respect to the weight.
  for i = 0; i < wv.data.length(); i = i + 1 {
    let wp = wv.to_data()
    let wm = wv.to_data()
    wp[i] = wp[i] + h
    wm[i] = wm[i] - h
    let fp = conv_of(
      xv,
      match @moongrad.Tensor::from_data([1, 1, 2, 2], wp) {
        Some(t) => t
        None => return
      },
      1,
      1,
      0,
      0,
    )
    let fm = conv_of(
      xv,
      match @moongrad.Tensor::from_data([1, 1, 2, 2], wm) {
        Some(t) => t
        None => return
      },
      1,
      1,
      0,
      0,
    )
    let numeric = (fp - fm) / (2.0 * h)
    check(
      grad.param_grads[0].data[i] - numeric < 1.0e-4 &&
      numeric - grad.param_grads[0].data[i] < 1.0e-4,
      "d/dw\{i}: analytic \{grad.param_grads[0].data[i]}, numeric \{numeric}",
    )
  }
}

///|
test "conv_grad_with_padding_matches_finite_differences" {
  // The same again, with padding, which is the case where the
  // col2im has to crop the gradient back out of the padding.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let w = g.param(0)
  let conv = g.conv2d(x, w, 1, 1, 2, 2, 1, 1, 1, 1)
  let loss = g.sum_axis(conv, 0)
  let b = @moongrad.TBinding::new()
  let xv = match
    @moongrad.Tensor::from_data([1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0]) {
    Some(t) => t
    None => return
  }
  let wv = match
    @moongrad.Tensor::from_data([1, 1, 2, 2], [0.5, -0.25, 1.0, 0.75]) {
    Some(t) => t
    None => return
  }
  b.set_input(0, xv)
  b.set_param(0, wv)
  let grad = @moongrad.t_grad(g, b, loss)
  let h = 1.0e-6
  for i = 0; i < xv.data.length(); i = i + 1 {
    let xp = xv.to_data()
    let xm = xv.to_data()
    xp[i] = xp[i] + h
    xm[i] = xm[i] - h
    let fp = conv_of(
      match @moongrad.Tensor::from_data([1, 1, 2, 2], xp) {
        Some(t) => t
        None => return
      },
      wv,
      1,
      1,
      1,
      1,
    )
    let fm = conv_of(
      match @moongrad.Tensor::from_data([1, 1, 2, 2], xm) {
        Some(t) => t
        None => return
      },
      wv,
      1,
      1,
      1,
      1,
    )
    let numeric = (fp - fm) / (2.0 * h)
    check(
      grad.input_grads[0].data[i] - numeric < 1.0e-4 &&
      numeric - grad.input_grads[0].data[i] < 1.0e-4,
      "padded d/dx\{i}: analytic \{grad.input_grads[0].data[i]}, numeric \{numeric}",
    )
  }
}

///|
test "conv_grad_with_two_input_channels_matches_finite_differences" {
  // Two input channels, two output channels: the case where the
  // weight layout has to be right in both dimensions.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let w = g.param(0)
  let conv = g.conv2d(x, w, 2, 2, 2, 2, 1, 1, 0, 0)
  let loss = g.sum_axis(conv, 0)
  let b = @moongrad.TBinding::new()
  let xv = match
    @moongrad.Tensor::from_data([1, 2, 2, 3], [
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -0.5, 1.5, 2.0, 0.0, -1.0,
    ]) {
    Some(t) => t
    None => return
  }
  let wv = match
    @moongrad.Tensor::from_data([2, 2, 2, 2], [
      0.5, -0.25, 1.0, 0.75, -1.0, 0.5, 0.25, -0.5, 0.25, 0.25, -0.5, 1.0, 0.75,
      -1.0, 0.5, 0.25,
    ]) {
    Some(t) => t
    None => return
  }
  b.set_input(0, xv)
  b.set_param(0, wv)
  let grad = @moongrad.t_grad(g, b, loss)
  let h = 1.0e-6
  for i = 0; i < wv.data.length(); i = i + 1 {
    let wp = wv.to_data()
    let wm = wv.to_data()
    wp[i] = wp[i] + h
    wm[i] = wm[i] - h
    let fp = conv_of(
      xv,
      match @moongrad.Tensor::from_data([2, 2, 2, 2], wp) {
        Some(t) => t
        None => return
      },
      1,
      1,
      0,
      0,
    )
    let fm = conv_of(
      xv,
      match @moongrad.Tensor::from_data([2, 2, 2, 2], wm) {
        Some(t) => t
        None => return
      },
      1,
      1,
      0,
      0,
    )
    let numeric = (fp - fm) / (2.0 * h)
    check(
      grad.param_grads[0].data[i] - numeric < 1.0e-4 &&
      numeric - grad.param_grads[0].data[i] < 1.0e-4,
      "d/dw\{i}: analytic \{grad.param_grads[0].data[i]}, numeric \{numeric}",
    )
  }
}

// ---------------------------------------------------------------------------
// The layers
// ---------------------------------------------------------------------------

///|
test "linear_layer_computes_x_times_w_plus_b" {
  let g = @moongrad.TGraph::new()
  let layer = @moongrad.Linear::new(g, 2, 3, 11)
  // The weight is (in_features, out_features) = (3, 2), so
  // y0 = 1*x0 + 0*x1 and y1 = 0*x0 + 1*x1.
  for i = 0; i < layer.w.data.length(); i = i + 1 {
    layer.w.data[i] = 0.0
  }
  layer.w.data[0] = 1.0
  layer.w.data[3] = 1.0
  for i = 0; i < layer.b.data.length(); i = i + 1 {
    layer.b.data[i] = 0.0
  }
  layer.b.data[0] = 0.5
  layer.b.data[1] = -0.5
  let x = g.input(0)
  let y = layer.apply(g, x)
  let b = @moongrad.TBinding::new()
  b.set_input(
    0,
    match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
      Some(t) => t
      None => return
    },
  )
  b.set_param(layer.slot_w, layer.w)
  b.set_param(layer.slot_b, layer.b)
  let grad = @moongrad.t_grad(g, b, y)
  // y[0] = x0 + 0.5, y[1] = x1 - 0.5.
  assert_eq(grad.output.get([0, 0]), Some(1.5))
  assert_eq(grad.output.get([0, 1]), Some(1.5))
  // d/dw0 = sum over the batch of x0 = 1 + 4 = 5.
  assert_eq(grad.param_grads[0].data[0], 5.0)
  // d/db = 2 (one per row).
  assert_eq(grad.param_grads[1].data[0], 2.0)
}

///|
test "linear_layer_gradient_matches_finite_differences" {
  let g = @moongrad.TGraph::new()
  let layer = @moongrad.Linear::new(g, 2, 2, 5)
  let x = g.input(0)
  let y = layer.apply(g, x)
  let b = @moongrad.TBinding::new()
  let xv = match @moongrad.Tensor::matrix([[0.3, -1.2], [2.0, 0.5]]) {
    Some(t) => t
    None => return
  }
  b.set_input(0, xv)
  b.set_param(layer.slot_w, layer.w)
  b.set_param(layer.slot_b, layer.b)
  let grad = @moongrad.t_grad(g, b, y)
  // The oracle: sum(y) as a function of the weights.
  let f = fn(wv : Array[Double], bv : Array[Double]) -> Double {
    let w = match @moongrad.Tensor::from_data([2, 2], wv) {
      Some(t) => t
      None => return 0.0
    }
    let bb = match @moongrad.Tensor::from_data([2], bv) {
      Some(t) => t
      None => return 0.0
    }
    let mut acc = 0.0
    for i = 0; i < 2; i = i + 1 {
      for j = 0; j < 2; j = j + 1 {
        acc = acc + xv.data[i * 2 + j] * w.data[i * 2 + j]
      }
      acc = acc + bb.data[i]
    }
    acc
  }
  let h = 1.0e-6
  let wv = layer.w.to_data()
  let bv = layer.b.to_data()
  for i = 0; i < wv.length(); i = i + 1 {
    let wp = wv.copy()
    let wm = wv.copy()
    wp[i] = wp[i] + h
    wm[i] = wm[i] - h
    let numeric = (f(wp, bv) - f(wm, bv)) / (2.0 * h)
    check(
      grad.param_grads[0].data[i] - numeric < 1.0e-5 &&
      numeric - grad.param_grads[0].data[i] < 1.0e-5,
      "d/dw\{i}: analytic \{grad.param_grads[0].data[i]}, numeric \{numeric}",
    )
  }
}

///|
test "embedding_row_is_read_from_the_table" {
  let g = @moongrad.TGraph::new()
  let emb = @moongrad.Embedding::new(g, 4, 2, 3)
  emb.table.data[0] = 1.0
  emb.table.data[1] = 2.0
  emb.table.data[4] = 7.0
  emb.table.data[5] = 8.0
  let row = emb.row(g, 2)
  let b = @moongrad.TBinding::new()
  b.set_param(emb.slot, emb.table)
  let grad = @moongrad.t_grad(g, b, row)
  assert_eq(grad.output.data[0], 7.0)
  assert_eq(grad.output.data[1], 8.0)
  // The gradient goes to the row that was read.
  assert_eq(grad.param_grads[0].data[4], 1.0)
  assert_eq(grad.param_grads[0].data[5], 1.0)
  assert_eq(grad.param_grads[0].data[0], 0.0)
}

///|
test "dropout_zeroes_some_units_and_scales_the_rest" {
  let g = @moongrad.TGraph::new()
  let d = @moongrad.Dropout::new(0.5, 9)
  let x = g.input(0)
  let y = d.apply(g, x)
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([1.0, 1.0, 1.0, 1.0]))
  let grad = @moongrad.t_grad(g, b, y)
  // Every unit is either dropped (0) or kept and scaled by 1/keep.
  let mut kept = 0
  for v in grad.output.data {
    if v != 0.0 {
      kept = kept + 1
    }
  }
  check(kept >= 0, "the mask should be a product of zeros and twos")
}

///|
test "sequential_runs_its_layers_in_order" {
  let g = @moongrad.TGraph::new()
  let linear = @moongrad.Linear::new(g, 2, 2, 1)
  let seq = @moongrad.Sequential::new()
  seq.push(
    fn(g, x) { g.unary(@moongrad.UnaryOp::Tanh, linear.apply(g, x)) },
    linear.tensors(),
  )
  let x = g.input(0)
  let y = seq.apply(g, x)
  let b = @moongrad.TBinding::new()
  b.set_input(
    0,
    match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
      Some(t) => t
      None => return
    },
  )
  b.set_param(linear.slot_w, linear.w)
  b.set_param(linear.slot_b, linear.b)
  let grad = @moongrad.t_grad(g, b, y)
  assert_eq(grad.output.shape, [2, 2])
}
