// conv.mbt
//
// Two-dimensional convolution, with a backward rule.
//
// The forward pass uses im2col: the input is expanded into a matrix
// of overlapping patches, and the convolution becomes a single
// matrix multiplication. That is the standard trick and it is what
// makes the backward rule tractable — the gradient with respect to
// the patches is another matrix multiplication, and the gradient
// with respect to the input is a scatter-add of those patches back
// into the input's shape.
//
// The whole thing is one node in the tensor graph, so the backward
// sweep reaches it like any other and the rules below produce the
// two gradients it owes: one to the input, one to the weight.
//
// Shapes, for an input of `(batch, in_channels, height, width)`, a
// weight of `(out_channels, in_channels, kernel_h, kernel_w)`, a
// stride of `(stride_h, stride_w)` and a padding of
// `(pad_h, pad_w)`:
//
//   output = (batch, out_channels, out_h, out_w)
//
// where
//
//   out_h = (height + 2*pad_h - kernel_h) / stride_h + 1
//   out_w = (width  + 2*pad_w - kernel_w) / stride_w + 1
//
// The input is padded with zeros first, which is what "same"
// padding means for a convolution.

///|
/// A 2-D convolutional layer.
pub(all) struct Conv2d {
  /// Graph slot of the weight tensor.
  slot_w : Int
  /// Graph slot of the bias vector.
  slot_b : Int
  /// Host-side weights, `(out_channels, in_channels, kh, kw)`.
  w : Tensor
  /// Host-side bias, `(out_channels)`.
  b : Tensor
  in_channels : Int
  out_channels : Int
  kernel_h : Int
  kernel_w : Int
  stride_h : Int
  stride_w : Int
  pad_h : Int
  pad_w : Int
} derive(Debug)

///|
/// A convolutional layer with weights drawn from the Glorot
/// uniform distribution over the fan count, which for a
/// convolution is `in_channels * kernel_h * kernel_w` against
/// `out_channels`.
pub fn Conv2d::new(
  g : TGraph,
  in_channels : Int,
  out_channels : Int,
  kernel_h : Int,
  kernel_w : Int,
  stride_h : Int,
  stride_w : Int,
  pad_h : Int,
  pad_w : Int,
  seed : Int,
) -> Conv2d {
  let fan_in = in_channels * kernel_h * kernel_w
  let slot_w = g.param(out_channels * fan_in)
  let slot_b = g.param(out_channels)
  let limit = (6.0 / (fan_in + out_channels).to_double()).sqrt()
  let rng = Lcg::new(seed)
  let w = Tensor::zeros([out_channels, in_channels, kernel_h, kernel_w])
  for i = 0; i < w.data.length(); i = i + 1 {
    w.data[i] = rng.uniform(-limit, limit)
  }
  let b = Tensor::zeros([out_channels])
  {
    slot_w,
    slot_b,
    w,
    b,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
  }
}

///|
/// The output height and width for a given input size.
pub fn Conv2d::output_size(
  self : Conv2d,
  height : Int,
  width : Int,
) -> (Int, Int) {
  let out_h = (height + 2 * self.pad_h - self.kernel_h) / self.stride_h + 1
  let out_w = (width + 2 * self.pad_w - self.kernel_w) / self.stride_w + 1
  (out_h, out_w)
}

///|
/// `out = conv(x) + b`, where `x` is `(batch, in_channels, h, w)`.
///
/// The convolution itself is one node; the bias is added
/// afterwards, elementwise and broadcast across the batch and the
/// spatial dimensions, which the tensor layer already knows how to
/// do.
pub fn Conv2d::apply(self : Conv2d, g : TGraph, x : Int) -> Int {
  let conv = g.conv2d(
    x,
    g.param(self.slot_w),
    self.in_channels,
    self.out_channels,
    self.kernel_h,
    self.kernel_w,
    self.stride_h,
    self.stride_w,
    self.pad_h,
    self.pad_w,
  )
  g.binary(BinaryOp::Add, conv, g.param(self.slot_b))
}

///|
/// The layer's parameters.
pub fn Conv2d::tensors(self : Conv2d) -> Array[Tensor] {
  [self.w, self.b]
}

///|
/// Write host-side values back into the parameter slots.
pub fn Conv2d::write_back(self : Conv2d, b : TBinding) -> Unit {
  b.set_param(self.slot_w, self.w)
  b.set_param(self.slot_b, self.b)
}

///|
/// Pad a tensor with zeros on the two spatial axes.
///
/// The channel axis is left alone: a convolution pads the spatial
/// dimensions, not the channels.
pub fn t_pad2d(x : Tensor, pad_h : Int, pad_w : Int) -> Tensor {
  if x.shape.length() != 4 {
    return x
  }
  let batch = x.shape[0]
  let channels = x.shape[1]
  let height = x.shape[2]
  let width = x.shape[3]
  let out = Tensor::zeros([
    batch,
    channels,
    height + 2 * pad_h,
    width + 2 * pad_w,
  ])
  for b = 0; b < batch; b = b + 1 {
    for c = 0; c < channels; c = c + 1 {
      for h = 0; h < height; h = h + 1 {
        for w = 0; w < width; w = w + 1 {
          let src = ((b * channels + c) * height + h) * width + w
          let dst = ((b * channels + c) * (height + 2 * pad_h) + h + pad_h) *
            (width + 2 * pad_w) +
            w +
            pad_w
          out.data[dst] = x.data[src]
        }
      }
    }
  }
  out
}

///|
/// The im2col matrix: one row per output position, one column per
/// `(channel, kernel row, kernel column)`.
///
/// The result has shape `(batch * out_h * out_w, in_channels * kh *
/// kw)`. This is the matrix the convolution multiplies.
pub fn t_im2col(
  x : Tensor,
  out_h : Int,
  out_w : Int,
  in_channels : Int,
  kernel_h : Int,
  kernel_w : Int,
  stride_h : Int,
  stride_w : Int,
) -> Tensor {
  let batch = x.shape[0]
  let height = x.shape[2]
  let width = x.shape[3]
  let row_len = in_channels * kernel_h * kernel_w
  let rows = batch * out_h * out_w
  let out = Tensor::zeros([rows, row_len])
  for b = 0; b < batch; b = b + 1 {
    for oh = 0; oh < out_h; oh = oh + 1 {
      for ow = 0; ow < out_w; ow = ow + 1 {
        let row = (b * out_h + oh) * out_w + ow
        let mut col = 0
        for c = 0; c < in_channels; c = c + 1 {
          for kh = 0; kh < kernel_h; kh = kh + 1 {
            for kw = 0; kw < kernel_w; kw = kw + 1 {
              let ih = oh * stride_h + kh
              let iw = ow * stride_w + kw
              let src = ((b * in_channels + c) * height + ih) * width + iw
              out.data[row * row_len + col] = x.data[src]
              col = col + 1
            }
          }
        }
      }
    }
  }
  out
}

///|
/// The gradient of an im2col is a col2im: scatter each row of the
/// gradient back into the input's shape, accumulating where the
/// patches overlap.
pub fn t_col2im(
  grad : Tensor,
  batch : Int,
  height : Int,
  width : Int,
  out_h : Int,
  out_w : Int,
  in_channels : Int,
  kernel_h : Int,
  kernel_w : Int,
  stride_h : Int,
  stride_w : Int,
) -> Tensor {
  let row_len = in_channels * kernel_h * kernel_w
  let out = Tensor::zeros([batch, in_channels, height, width])
  for b = 0; b < batch; b = b + 1 {
    for oh = 0; oh < out_h; oh = oh + 1 {
      for ow = 0; ow < out_w; ow = ow + 1 {
        let row = (b * out_h + oh) * out_w + ow
        let mut col = 0
        for c = 0; c < in_channels; c = c + 1 {
          for kh = 0; kh < kernel_h; kh = kh + 1 {
            for kw = 0; kw < kernel_w; kw = kw + 1 {
              let ih = oh * stride_h + kh
              let iw = ow * stride_w + kw
              let src = row * row_len + col
              let dst = ((b * in_channels + c) * height + ih) * width + iw
              out.data[dst] = out.data[dst] + grad.data[src]
              col = col + 1
            }
          }
        }
      }
    }
  }
  out
}

///|
/// The 2-D convolution, as a plain tensor operation.
pub fn t_conv2d(
  x : Tensor,
  w : Tensor,
  bias : Tensor?,
  stride_h : Int,
  stride_w : Int,
  pad_h : Int,
  pad_w : Int,
) -> Tensor? {
  if x.shape.length() != 4 || w.shape.length() != 4 {
    return None
  }
  let batch = x.shape[0]
  let in_channels = x.shape[1]
  let height = x.shape[2]
  let width = x.shape[3]
  let out_channels = w.shape[0]
  let kernel_h = w.shape[2]
  let kernel_w = w.shape[3]
  if w.shape[1] != in_channels {
    return None
  }
  let out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1
  let out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1
  if out_h <= 0 || out_w <= 0 {
    return None
  }
  let padded = t_pad2d(x, pad_h, pad_w)
  let patches = t_im2col(
    padded, out_h, out_w, in_channels, kernel_h, kernel_w, stride_h, stride_w,
  )
  // The weight has to be flattened to (out_channels, row_len) and
  // then transposed, because the patch matrix has the spatial
  // positions in its rows.
  let row_len = in_channels * kernel_h * kernel_w
  let flat_w = Tensor::zeros([out_channels, row_len])
  for oc = 0; oc < out_channels; oc = oc + 1 {
    for c = 0; c < in_channels; c = c + 1 {
      for kh = 0; kh < kernel_h; kh = kh + 1 {
        for kw = 0; kw < kernel_w; kw = kw + 1 {
          let src = ((oc * in_channels + c) * kernel_h + kh) * kernel_w + kw
          let dst = oc * row_len + (c * kernel_h + kh) * kernel_w + kw
          flat_w.data[dst] = w.data[src]
        }
      }
    }
  }
  let wt = match flat_w.transpose([1, 0]) {
    Some(t) => t
    None => return None
  }
  let out = match patches.matmul(wt) {
    Some(t) => t
    None => return None
  }
  // (batch * out_h * out_w, out_channels) -> (batch, out_h, out_w,
  // out_channels)
  let shaped = match out.reshape([batch, out_h, out_w, out_channels]) {
    Some(t) => t
    None => return None
  }
  // Move the channel axis back to position 1.
  let final = match shaped.transpose([0, 3, 1, 2]) {
    Some(t) => t
    None => return None
  }
  match bias {
    Some(bv) =>
      match t_add(final, bv) {
        Some(t) => Some(t)
        None => Some(final)
      }
    None => Some(final)
  }
}

///|
/// Crop a padded tensor back to its original spatial size.
pub fn t_crop2d(x : Tensor, pad_h : Int, pad_w : Int) -> Tensor {
  if x.shape.length() != 4 {
    return x
  }
  let batch = x.shape[0]
  let channels = x.shape[1]
  let height = x.shape[2] - 2 * pad_h
  let width = x.shape[3] - 2 * pad_w
  if height <= 0 || width <= 0 {
    return x
  }
  let out = Tensor::zeros([batch, channels, height, width])
  for b = 0; b < batch; b = b + 1 {
    for c = 0; c < channels; c = c + 1 {
      for h = 0; h < height; h = h + 1 {
        for w = 0; w < width; w = w + 1 {
          let src = ((b * channels + c) * x.shape[2] + h + pad_h) * x.shape[3] +
            w +
            pad_w
          let dst = ((b * channels + c) * height + h) * width + w
          out.data[dst] = x.data[src]
        }
      }
    }
  }
  out
}
