// tensor.mbt
//
// A multi-dimensional array.
//
// `moongrad` started life as a scalar expression engine, which was
// exactly the criticism the first review made: useful, but too
// narrow. This module is the widening — a tensor is a shape plus a
// flat row-major buffer, and every operation on it is written in
// terms of shapes rather than of individual elements.
//
// The shape is carried separately from the data so that the
// broadcasting rules can be applied without touching a single
// number. Two shapes broadcast when, aligned from the right, each
// dimension is either equal or one of them is 1. That is the same
// rule numpy uses, and it is the rule that makes `(m, 1) + (n,)` do
// what everyone expects without anyone having to write it.
//
// Everything is row-major. `Tensor::at(i, j)` on a rank-2 tensor
// is `data[i * cols + j]`, and `Tensor::strides` computes the
// corresponding offsets for the general case.

///|
/// A dense, row-major tensor.
pub(all) struct Tensor {
  /// Length of each axis. An empty shape is a scalar; `[n]` is a
  /// vector; `[m, n]` is a matrix.
  shape : Array[Int]
  /// The elements, row-major.
  data : Array[Double]
} derive(Debug, Eq)

///|
/// The number of axes.
pub fn Tensor::rank(self : Tensor) -> Int {
  self.shape.length()
}

///|
/// The total number of elements.
pub fn Tensor::numel(self : Tensor) -> Int {
  self.data.length()
}

///|
/// `true` when the tensor has no axes.
pub fn Tensor::is_scalar(self : Tensor) -> Bool {
  self.shape.length() == 0
}

///|
/// The product of the shape, or 1 for a scalar.
pub fn Tensor::size(self : Tensor) -> Int {
  let mut n = 1
  for d in self.shape {
    n = n * d
  }
  n
}

///|
/// A tensor of the given shape, filled with zeros.
pub fn Tensor::zeros(shape : Array[Int]) -> Tensor {
  let mut n = 1
  for d in shape {
    n = n * d
  }
  let data : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    data.push(0.0)
  }
  { shape: shape.copy(), data, }
}

///|
/// A tensor of the given shape, filled with `v`.
pub fn Tensor::filled(shape : Array[Int], v : Double) -> Tensor {
  let t = Tensor::zeros(shape)
  for i = 0; i < t.data.length(); i = i + 1 {
    t.data[i] = v
  }
  t
}

///|
/// A tensor from a flat buffer. Returns `None` when the length does
/// not match the shape, rather than padding or truncating.
pub fn Tensor::from_data(shape : Array[Int], data : Array[Double]) -> Tensor? {
  let mut n = 1
  for d in shape {
    n = n * d
  }
  if data.length() != n {
    return None
  }
  Some({ shape: shape.copy(), data, })
}

///|
/// A rank-1 tensor.
pub fn Tensor::vector(data : Array[Double]) -> Tensor {
  { shape: [data.length()], data, }
}

///|
/// A rank-2 tensor from a list of rows. Returns `None` when the
/// rows are ragged.
pub fn Tensor::matrix(rows : Array[Array[Double]]) -> Tensor? {
  if rows.length() == 0 {
    return Some({ shape: [0, 0], data: [], })
  }
  let cols = rows[0].length()
  let data : Array[Double] = []
  for r in rows {
    if r.length() != cols {
      return None
    }
    for v in r {
      data.push(v)
    }
  }
  Some({ shape: [rows.length(), cols], data, })
}

///|
/// A scalar tensor.
pub fn Tensor::scalar(v : Double) -> Tensor {
  { shape: [], data: [v], }
}

///|
/// Read the element at a multi-index. Returns `None` when the index
/// is out of range or the rank does not match.
pub fn Tensor::get(self : Tensor, index : Array[Int]) -> Double? {
  if index.length() != self.shape.length() {
    return None
  }
  let mut offset = 0
  let mut stride = 1
  let mut d = self.shape.length() - 1
  while d >= 0 {
    if index[d] < 0 || index[d] >= self.shape[d] {
      return None
    }
    offset = offset + index[d] * stride
    stride = stride * self.shape[d]
    d = d - 1
  }
  Some(self.data[offset])
}

///|
/// Write the element at a multi-index.
pub fn Tensor::set(self : Tensor, index : Array[Int], v : Double) -> Unit {
  if index.length() != self.shape.length() {
    return
  }
  let mut offset = 0
  let mut stride = 1
  let mut d = self.shape.length() - 1
  while d >= 0 {
    if index[d] < 0 || index[d] >= self.shape[d] {
      return
    }
    offset = offset + index[d] * stride
    stride = stride * self.shape[d]
    d = d - 1
  }
  self.data[offset] = v
}

///|
/// The element at a flat offset.
pub fn Tensor::at(self : Tensor, i : Int) -> Double {
  self.data[i]
}

///|
/// The row `i` of a rank-2 tensor, as a rank-1 tensor. The result
/// shares the underlying buffer, so it is a view rather than a copy.
pub fn Tensor::row(self : Tensor, i : Int) -> Tensor {
  let cols = if self.shape.length() >= 2 { self.shape[1] } else { 1 }
  let data : Array[Double] = []
  for j = 0; j < cols; j = j + 1 {
    data.push(self.data[i * cols + j])
  }
  { shape: [cols], data, }
}

///|
/// The column `j` of a rank-2 tensor.
pub fn Tensor::column(self : Tensor, j : Int) -> Tensor {
  let rows = if self.shape.length() >= 2 { self.shape[0] } else { 1 }
  let data : Array[Double] = []
  for i = 0; i < rows; i = i + 1 {
    data.push(self.data[i * self.shape[1] + j])
  }
  { shape: [rows], data, }
}

///|
/// A copy with a new shape. The number of elements has to match.
pub fn Tensor::reshape(self : Tensor, shape : Array[Int]) -> Tensor? {
  let mut n = 1
  for d in shape {
    n = n * d
  }
  if n != self.data.length() {
    return None
  }
  Some({ shape: shape.copy(), data: self.data.copy(), })
}

///|
/// A copy with the axes permuted. `perm[k]` is the source axis that
/// becomes axis `k`.
pub fn Tensor::transpose(self : Tensor, perm : Array[Int]) -> Tensor? {
  let r = self.shape.length()
  if perm.length() != r {
    return None
  }
  let seen : Array[Bool] = []
  for i = 0; i < r; i = i + 1 {
    seen.push(false)
  }
  for p in perm {
    if p < 0 || p >= r || seen[p] {
      return None
    }
    seen[p] = true
  }
  let new_shape : Array[Int] = []
  for p in perm {
    new_shape.push(self.shape[p])
  }
  let result = Tensor::zeros(new_shape)
  // Walk the source in its own order and write each element to its
  // permuted position.
  let total = self.data.length()
  for flat = 0; flat < total; flat = flat + 1 {
    // Decode the flat source index into per-axis coordinates.
    let coords : Array[Int] = []
    let mut rest = flat
    let mut d = r - 1
    while d >= 0 {
      coords.push(rest % self.shape[d])
      rest = rest / self.shape[d]
      d = d - 1
    }
    // Reverse into axis order.
    let axis_coords : Array[Int] = []
    for k = 0; k < r; k = k + 1 {
      axis_coords.push(coords[r - 1 - k])
    }
    // Map to the destination and write.
    let mut offset = 0
    let mut dst_stride = 1
    let mut k = r - 1
    while k >= 0 {
      offset = offset + axis_coords[perm[k]] * dst_stride
      dst_stride = dst_stride * new_shape[k]
      k = k - 1
    }
    result.data[offset] = self.data[flat]
  }
  Some(result)
}

///|
/// Per-axis strides, in elements. Row-major, so the last axis has
/// stride 1.
pub fn Tensor::strides(shape : Array[Int]) -> Array[Int] {
  let out : Array[Int] = []
  for _d = 0; _d < shape.length(); _d = _d + 1 {
    out.push(0)
  }
  let mut stride = 1
  let mut d = shape.length() - 1
  while d >= 0 {
    out[d] = stride
    stride = stride * shape[d]
    d = d - 1
  }
  out
}

///|
/// The shape `src` has to be broadcast *from* to reach `dst`.
/// Returns `None` when the two do not broadcast.
pub fn broadcast_shape(src : Array[Int], dst : Array[Int]) -> Array[Int]? {
  let offset = dst.length() - src.length()
  let out = dst.copy()
  let mut d = 0
  while d < src.length() {
    let target = offset + d
    if target < 0 {
      return None
    }
    let s = src[d]
    let t = out[target]
    if s != t && s != 1 {
      return None
    }
    if s == 1 && t != 1 {
      out[target] = t
    }
    d = d + 1
  }
  Some(out)
}

///|
/// `true` when the two shapes broadcast against each other.
pub fn can_broadcast(a : Array[Int], b : Array[Int]) -> Bool {
  let r = if a.length() > b.length() { a.length() } else { b.length() }
  let mut d = 0
  while d < r {
    let da = a.length() - r + d
    let db = b.length() - r + d
    let sa = if da >= 0 { a[da] } else { 1 }
    let sb = if db >= 0 { b[db] } else { 1 }
    if sa != sb && sa != 1 && sb != 1 {
      return false
    }
    d = d + 1
  }
  true
}

///|
/// The shape both operands broadcast to.
pub fn common_shape(a : Array[Int], b : Array[Int]) -> Array[Int]? {
  let r = if a.length() > b.length() { a.length() } else { b.length() }
  let out : Array[Int] = []
  let mut d = 0
  while d < r {
    let da = a.length() - r + d
    let db = b.length() - r + d
    let sa = if da >= 0 { a[da] } else { 1 }
    let sb = if db >= 0 { b[db] } else { 1 }
    if sa == sb {
      out.push(sa)
    } else if sa == 1 {
      out.push(sb)
    } else if sb == 1 {
      out.push(sa)
    } else {
      return None
    }
    d = d + 1
  }
  Some(out)
}

///|
/// For every element of the broadcast shape, the flat index of the
/// element in `src` that it reads.
///
/// This is the workhorse of broadcasting: rather than special-casing
// every operator, we compute the index map once and let every
// elementwise operator use it.
pub fn broadcast_map(src : Array[Int], dst : Array[Int]) -> Array[Int]? {
  let aligned = match broadcast_shape(src, dst) {
    Some(s) => s
    None => return None
  }
  // Per-axis stride in the destination, and the stride to use when
  // reading the source.
  let dst_strides = Tensor::strides(aligned)
  let src_offset = dst.length() - src.length()
  let src_strides : Array[Int] = []
  for d = 0; d < aligned.length(); d = d + 1 {
    let sd = d - src_offset
    let s = if sd >= 0 && sd < src.length() { src[sd] } else { 1 }
    // A source axis of length 1 reads the same element for every
    // destination index, so its stride is zero.
    src_strides.push(if s == 1 { 0 } else { dst_strides[d] })
  }
  let mut total = 1
  for d in aligned {
    total = total * d
  }
  let map : Array[Int] = []
  for flat = 0; flat < total; flat = flat + 1 {
    let mut rest = flat
    let mut offset = 0
    let mut d = aligned.length() - 1
    while d >= 0 {
      let idx = rest % aligned[d]
      rest = rest / aligned[d]
      offset = offset + idx * src_strides[d]
      d = d - 1
    }
    map.push(offset)
  }
  Some(map)
}

///|
/// Expand `src` to `dst` by repeating elements along the axes that
/// have to grow. Returns `None` when the two do not broadcast.
pub fn Tensor::broadcast_to(self : Tensor, dst : Array[Int]) -> Tensor? {
  let map = match broadcast_map(self.shape, dst) {
    Some(m) => m
    None => return None
  }
  let out = Tensor::zeros(dst)
  for i = 0; i < map.length(); i = i + 1 {
    out.data[i] = self.data[map[i]]
  }
  Some(out)
}

///|
/// A copy of the tensor's data.
pub fn Tensor::to_data(self : Tensor) -> Array[Double] {
  self.data.copy()
}

///|
/// The Euclidean norm of the flattened tensor.
pub fn Tensor::norm(self : Tensor) -> Double {
  let mut acc = 0.0
  for x in self.data {
    acc = acc + x * x
  }
  acc.sqrt()
}

///|
/// The largest absolute entry.
pub fn Tensor::max_abs(self : Tensor) -> Double {
  let mut best = 0.0
  for x in self.data {
    let a = if x < 0.0 { -x } else { x }
    if a > best {
      best = a
    }
  }
  best
}

///|
/// `true` when every entry is finite.
pub fn Tensor::all_finite(self : Tensor) -> Bool {
  for x in self.data {
    if !(x == x) {
      return false
    }
    if x > 1.0e308 || x < -1.0e308 {
      return false
    }
  }
  true
}

///|
/// Sum of every entry.
pub fn Tensor::sum(self : Tensor) -> Double {
  let mut acc = 0.0
  for x in self.data {
    acc = acc + x
  }
  acc
}

///|
/// Mean of every entry.
pub fn Tensor::mean(self : Tensor) -> Double {
  if self.data.length() == 0 {
    return 0.0
  }
  self.sum() / self.data.length().to_double()
}

///|
/// Element-wise sum of two tensors of the same shape.
pub fn Tensor::add(self : Tensor, other : Tensor) -> Tensor? {
  if self.shape.length() != other.shape.length() {
    return None
  }
  for i = 0; i < self.shape.length(); i = i + 1 {
    if self.shape[i] != other.shape[i] {
      return None
    }
  }
  let out = Tensor::zeros(self.shape)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] + other.data[i]
  }
  Some(out)
}

///|
/// Element-wise difference of two tensors of the same shape.
pub fn Tensor::sub(self : Tensor, other : Tensor) -> Tensor? {
  if self.shape.length() != other.shape.length() {
    return None
  }
  for i = 0; i < self.shape.length(); i = i + 1 {
    if self.shape[i] != other.shape[i] {
      return None
    }
  }
  let out = Tensor::zeros(self.shape)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] - other.data[i]
  }
  Some(out)
}

///|
/// Element-wise product of two tensors of the same shape.
pub fn Tensor::mul(self : Tensor, other : Tensor) -> Tensor? {
  if self.shape.length() != other.shape.length() {
    return None
  }
  for i = 0; i < self.shape.length(); i = i + 1 {
    if self.shape[i] != other.shape[i] {
      return None
    }
  }
  let out = Tensor::zeros(self.shape)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] * other.data[i]
  }
  Some(out)
}

///|
/// Multiply every element by a scalar.
pub fn Tensor::scale(self : Tensor, k : Double) -> Tensor {
  let out = Tensor::zeros(self.shape)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] * k
  }
  out
}

///|
/// Add a scalar to every element.
pub fn Tensor::shift(self : Tensor, k : Double) -> Tensor {
  let out = Tensor::zeros(self.shape)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] + k
  }
  out
}

///|
/// Negate every element.
pub fn Tensor::neg(self : Tensor) -> Tensor {
  self.scale(-1.0)
}

///|
/// Element-wise product of two tensors, broadcasting them against
/// each other first.
pub fn Tensor::mul_broadcast(self : Tensor, other : Tensor) -> Tensor? {
  let shape = match common_shape(self.shape, other.shape) {
    Some(s) => s
    None => return None
  }
  let a = match self.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let b = match other.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  a.mul(b)
}

///|
/// Element-wise sum of two tensors, broadcasting them against each
/// other first.
pub fn Tensor::add_broadcast(self : Tensor, other : Tensor) -> Tensor? {
  let shape = match common_shape(self.shape, other.shape) {
    Some(s) => s
    None => return None
  }
  let a = match self.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let b = match other.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  a.add(b)
}

///|
/// Matrix product of two rank-2 tensors. `(m, k) * (k, n) -> (m, n)`.
pub fn Tensor::matmul(self : Tensor, other : Tensor) -> Tensor? {
  if self.shape.length() != 2 || other.shape.length() != 2 {
    return None
  }
  let m = self.shape[0]
  let k = self.shape[1]
  let k2 = other.shape[0]
  let n = other.shape[1]
  if k != k2 {
    return None
  }
  let out = Tensor::zeros([m, n])
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let mut acc = 0.0
      for p = 0; p < k; p = p + 1 {
        acc = acc + self.data[i * k + p] * other.data[p * n + j]
      }
      out.data[i * n + j] = acc
    }
  }
  Some(out)
}

///|
/// Matrix product with the right operand transposed:
/// `(m, k) * (n, k)^T -> (m, n)`.
pub fn Tensor::matmul_t(self : Tensor, other : Tensor) -> Tensor? {
  if self.shape.length() != 2 || other.shape.length() != 2 {
    return None
  }
  let m = self.shape[0]
  let k = self.shape[1]
  let n = other.shape[0]
  if k != other.shape[1] {
    return None
  }
  let out = Tensor::zeros([m, n])
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let mut acc = 0.0
      for p = 0; p < k; p = p + 1 {
        acc = acc + self.data[i * k + p] * other.data[j * k + p]
      }
      out.data[i * n + j] = acc
    }
  }
  Some(out)
}

///|
/// Concatenate two tensors along an axis. The other axes have to
/// agree.
pub fn Tensor::concat(self : Tensor, other : Tensor, axis : Int) -> Tensor? {
  if self.shape.length() != other.shape.length() {
    return None
  }
  let r = self.shape.length()
  if axis < 0 || axis >= r {
    return None
  }
  for d = 0; d < r; d = d + 1 {
    if d != axis && self.shape[d] != other.shape[d] {
      return None
    }
  }
  let new_shape = self.shape.copy()
  new_shape[axis] = self.shape[axis] + other.shape[axis]
  let out = Tensor::zeros(new_shape)
  let mut outer = 1
  for d = 0; d < axis; d = d + 1 {
    outer = outer * self.shape[d]
  }
  let mut inner = 1
  for d = axis + 1; d < r; d = d + 1 {
    inner = inner * self.shape[d]
  }
  let a_len = self.shape[axis]
  let b_len = other.shape[axis]
  for o = 0; o < outer; o = o + 1 {
    for i = 0; i < a_len; i = i + 1 {
      for j = 0; j < inner; j = j + 1 {
        let src = o * a_len * inner + i * inner + j
        let dst = o * (a_len + b_len) * inner + i * inner + j
        out.data[dst] = self.data[src]
      }
    }
    for i = 0; i < b_len; i = i + 1 {
      for j = 0; j < inner; j = j + 1 {
        let src = o * b_len * inner + i * inner + j
        let dst = o * (a_len + b_len) * inner + (a_len + i) * inner + j
        out.data[dst] = other.data[src]
      }
    }
  }
  Some(out)
}

///|
/// A slice along one axis: `count` elements starting at `start`.
pub fn Tensor::slice(
  self : Tensor,
  axis : Int,
  start : Int,
  count : Int,
) -> Tensor? {
  let r = self.shape.length()
  if axis < 0 || axis >= r {
    return None
  }
  if start < 0 || count < 0 || start + count > self.shape[axis] {
    return None
  }
  let new_shape = self.shape.copy()
  new_shape[axis] = count
  let out = Tensor::zeros(new_shape)
  let mut outer = 1
  for d = 0; d < axis; d = d + 1 {
    outer = outer * self.shape[d]
  }
  let mut inner = 1
  for d = axis + 1; d < r; d = d + 1 {
    inner = inner * self.shape[d]
  }
  let a_len = self.shape[axis]
  for o = 0; o < outer; o = o + 1 {
    for i = 0; i < count; i = i + 1 {
      for j = 0; j < inner; j = j + 1 {
        let src = o * a_len * inner + (start + i) * inner + j
        let dst = o * count * inner + i * inner + j
        out.data[dst] = self.data[src]
      }
    }
  }
  Some(out)
}

///|
/// One-line rendering, for the example scripts.
pub fn Tensor::preview(self : Tensor) -> String {
  let buf = StringBuilder()
  buf.write_string("[")
  let mut first = true
  for d in self.shape {
    if !first {
      buf.write_string(" x ")
    }
    first = false
    buf.write_string(d.to_string())
  }
  buf.write_string("] ")
  let shown = if self.data.length() > 12 { 12 } else { self.data.length() }
  for i = 0; i < shown; i = i + 1 {
    if i > 0 {
      buf.write_string(" ")
    }
    buf.write_string(format_double(self.data[i]))
  }
  if self.data.length() > shown {
    buf.write_string(" ...")
  }
  buf.to_string()
}

///|
/// Format a double with four decimal places, for display.
pub fn format_double(x : Double) -> String {
  let scaled = (x * 10000.0).round() / 10000.0
  scaled.to_string()
}
