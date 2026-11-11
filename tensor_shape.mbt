///|
/// Take a window out of a tensor.
///
/// The window starts at `start` along `axis` and runs for the
/// target's length there; the other axes are taken whole. This is
/// the inverse of `t_pad_to`: padding puts a tensor into a larger
/// one, cropping takes it back out.
pub fn t_crop_to(
  x : Tensor,
  shape : Array[Int32],
  axis : Int32,
  start : Int32,
) -> Tensor? {
  if x.shape.length() != shape.length() {
    return None
  }
  if axis < 0 || axis >= shape.length() {
    return None
  }
  for d = 0; d < shape.length(); d = d + 1 {
    if d != axis && x.shape[d] != shape[d] {
      return None
    }
  }
  let len = shape[axis]
  if start < 0 || start + len > x.shape[axis] {
    return None
  }
  let out = Tensor::zeros(shape)
  let mut outer = 1
  for d = 0; d < axis; d = d + 1 {
    outer = outer * x.shape[d]
  }
  let mut inner = 1
  for d = axis + 1; d < x.shape.length(); d = d + 1 {
    inner = inner * x.shape[d]
  }
  for o = 0; o < outer; o = o + 1 {
    for i = 0; i < len; i = i + 1 {
      for j = 0; j < inner; j = j + 1 {
        let src = o * x.shape[axis] * inner + (start + i) * inner + j
        let dst = o * len * inner + i * inner + j
        out.data[dst] = x.data[src]
      }
    }
  }
  Some(out)
}

///|
/// Place a row of values into a larger tensor.
///
/// Row `index` of the result holds `x`'s values; everything else is
/// zero. The row is the `index`-th group of `dim` consecutive
/// elements, which is the layout a parameter vector read assumes.
pub fn t_scatter_row(
  x : Tensor,
  shape : Array[Int32],
  index : Int32,
  dim : Int32,
) -> Tensor? {
  if x.data.length() != dim {
    return None
  }
  let out = Tensor::zeros(shape)
  let start = index * dim
  if start < 0 || start + dim > out.data.length() {
    return None
  }
  for i = 0; i < dim; i = i + 1 {
    out.data[start + i] = x.data[i]
  }
  Some(out)
}
