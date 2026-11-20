// tensor_gather.mbt
//
// Gathering rows and putting them back.
//
// A gather takes rows of a tensor by index; a scatter puts them back.
// The two are inverses of each other, which is what makes the pair
// useful for anything that has to be differentiated: the gradient
// of a gather is a scatter, and the gradient of a scatter is a
// gather.
//
// The index is a node of its own rather than a number baked into the
// graph, so the same compiled program can gather different rows at
// different bindings. That is what makes it usable for a lookup
// table whose index comes from the data.
//
// An index that is out of range, or that is not a whole number, is
// refused. A gather that quietly dropped rows would change the size
// of the result, and a caller indexing into the result would not know
// which rows it had lost.

///|
/// The number of elements in one row of a tensor.
fn t_row_length(t : Tensor) -> Int32 {
  let mut n = 1
  for i = 1; i < t.shape.length(); i = i + 1 {
    n = n * t.shape[i]
  }
  n
}

///|
/// The indices a gather or a scatter is given, as whole numbers.
///
/// A scalar index counts as one index. An index that is not a whole
/// number is refused rather than rounded: rounding would quietly pick
/// a row the caller did not ask for.
fn t_gather_indices(index : Tensor) -> Array[Int32]? {
  let out : Array[Int32] = []
  if index.shape.length() == 0 {
    if index.data.length() != 1 {
      return None
    }
    let v = index.data[0]
    if v != v || v != v.round() {
      return None
    }
    out.push(v.to_int())
    return Some(out)
  }
  if index.shape.length() != 1 {
    return None
  }
  for i = 0; i < index.data.length(); i = i + 1 {
    let v = index.data[i]
    if v != v || v != v.round() {
      return None
    }
    out.push(v.to_int())
  }
  Some(out)
}

///|
/// Take rows of `x` by index, along the first axis.
///
/// The result has the shape of `x` with its first axis replaced by
/// the number of indices.
pub fn t_gather_rows(x : Tensor, index : Tensor) -> Tensor? {
  if x.shape.length() == 0 {
    // A scalar has no rows to take.
    return None
  }
  let indices = match t_gather_indices(index) {
    Some(v) => v
    None => return None
  }
  let rows = x.shape[0]
  let width = t_row_length(x)
  for i = 0; i < indices.length(); i = i + 1 {
    if indices[i] < 0 || indices[i] >= rows {
      // An index out of range is refused: dropping the row would
      // change the size of the result.
      return None
    }
  }
  let out_shape : Array[Int32] = [indices.length()]
  for i = 1; i < x.shape.length(); i = i + 1 {
    out_shape.push(x.shape[i])
  }
  let out = Tensor::zeros(out_shape)
  for i = 0; i < indices.length(); i = i + 1 {
    let from = indices[i] * width
    let to = i * width
    for j = 0; j < width; j = j + 1 {
      out.data[to + j] = x.data[from + j]
    }
  }
  Some(out)
}

///|
/// Put rows back where they came from, along the first axis.
///
/// Rows that several indices point at are summed. `shape` is the
/// shape of the tensor the rows came from; it is given here because
/// the rows themselves do not carry it.
pub fn t_scatter_rows(
  x : Tensor,
  index : Tensor,
  shape : Array[Int32],
) -> Tensor? {
  if x.shape.length() == 0 || shape.length() == 0 {
    return None
  }
  let indices = match t_gather_indices(index) {
    Some(v) => v
    None => return None
  }
  let width = t_row_length(x)
  let rows = shape[0]
  // The width of a row of the target.
  let mut target_width = 1
  for i = 1; i < shape.length(); i = i + 1 {
    target_width = target_width * shape[i]
  }
  if width != target_width {
    return None
  }
  let out = Tensor::zeros(shape)
  for i = 0; i < indices.length(); i = i + 1 {
    if indices[i] < 0 || indices[i] >= rows {
      return None
    }
    let from = i * width
    let to = indices[i] * width
    for j = 0; j < width; j = j + 1 {
      out.data[to + j] = out.data[to + j] + x.data[from + j]
    }
  }
  Some(out)
}
