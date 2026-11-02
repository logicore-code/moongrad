// tensor.mbt
//
// A row-major 2-D matrix of doubles, and a handful of the linear
// algebra the examples need.
//
// Deliberately small. The interesting machinery in this project is
// the differentiation, not the linear algebra, and a tensor library
// with strides and broadcasting and lazy evaluation would bury it.
// What a gradient engine actually needs from a matrix type is: a
// flat backing store, a shape, and enough indexing to write
// `C[i][j] = sum_k A[i][k] * B[k][j]`. That is what this is.
//
// Matrices are *not* differentiable values. They are host-side
// containers whose entries are read into the graph through
// `ParamGet`, one scalar node per element. The graph therefore sees
// scalars, which keeps the IR in node.mbt simple, and the matrix
// keeps the host code readable.

///|
/// An `rows x cols` matrix, row-major.
pub(all) struct Matrix {
  rows : Int
  cols : Int
  data : Array[Double]
} derive(Debug)

///|
pub extend Matrix with Debug::{to_repr}

///|
/// A `rows x cols` matrix of zeros.
pub fn Matrix::zeros(rows : Int, cols : Int) -> Matrix {
  let data : Array[Double] = []
  for i = 0; i < rows * cols; i = i + 1 {
    data.push(0.0)
  }
  { rows, cols, data, }
}

///|
/// A `rows x cols` matrix with every entry `v`.
pub fn Matrix::filled(rows : Int, cols : Int, v : Double) -> Matrix {
  let m = Matrix::zeros(rows, cols)
  for i = 0; i < m.data.length(); i = i + 1 {
    m.data[i] = v
  }
  m
}

///|
/// The `n x n` identity.
pub fn Matrix::identity(n : Int) -> Matrix {
  let m = Matrix::zeros(n, n)
  for i = 0; i < n; i = i + 1 {
    m.set(i, i, 1.0)
  }
  m
}

///|
/// A matrix from a flat row-major array. Returns `None` when the
/// length does not match the shape, rather than padding or
/// truncating: a silent reshape is the kind of thing that turns into
/// a week of debugging.
pub fn Matrix::from_array(
  rows : Int,
  cols : Int,
  data : Array[Double],
) -> Matrix? {
  if data.length() != rows * cols {
    return None
  }
  Some({ rows, cols, data, })
}

///|
/// A column vector built from a flat array.
pub fn Matrix::column(xs : Array[Double]) -> Matrix {
  let m = Matrix::zeros(xs.length(), 1)
  for i = 0; i < xs.length(); i = i + 1 {
    m.set(i, 0, xs[i])
  }
  m
}

///|
/// Element `(row, col)`.
pub fn Matrix::get(self : Matrix, row : Int, col : Int) -> Double {
  self.data[row * self.cols + col]
}

///|
/// Write `v` at `(row, col)`.
pub fn Matrix::set(self : Matrix, row : Int, col : Int, v : Double) -> Unit {
  self.data[row * self.cols + col] = v
}

///|
/// Number of scalars in the matrix.
pub fn Matrix::len(self : Matrix) -> Int {
  self.data.length()
}

///|
/// Element-wise sum.
pub fn Matrix::add(self : Matrix, other : Matrix) -> Matrix {
  let out = Matrix::zeros(self.rows, self.cols)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] + other.data[i]
  }
  out
}

///|
/// Element-wise difference.
pub fn Matrix::sub(self : Matrix, other : Matrix) -> Matrix {
  let out = Matrix::zeros(self.rows, self.cols)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] - other.data[i]
  }
  out
}

///|
/// Scalar multiple.
pub fn Matrix::scale(self : Matrix, k : Double) -> Matrix {
  let out = Matrix::zeros(self.rows, self.cols)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] * k
  }
  out
}

///|
/// Matrix product. Returns `None` on a shape mismatch.
pub fn Matrix::mul(self : Matrix, other : Matrix) -> Matrix? {
  if self.cols != other.rows {
    return None
  }
  let out = Matrix::zeros(self.rows, other.cols)
  for i = 0; i < self.rows; i = i + 1 {
    for j = 0; j < other.cols; j = j + 1 {
      let mut acc = 0.0
      for k = 0; k < self.cols; k = k + 1 {
        acc = acc + self.get(i, k) * other.get(k, j)
      }
      out.set(i, j, acc)
    }
  }
  Some(out)
}

///|
/// Matrix product with a transposed right operand: `self * other^T`.
/// Written out because the forward pass and the backward pass both
/// need it and both need it to be cheap.
pub fn Matrix::mul_t(self : Matrix, other : Matrix) -> Matrix? {
  if self.cols != other.cols {
    return None
  }
  let out = Matrix::zeros(self.rows, other.rows)
  for i = 0; i < self.rows; i = i + 1 {
    for j = 0; j < other.rows; j = j + 1 {
      let mut acc = 0.0
      for k = 0; k < self.cols; k = k + 1 {
        acc = acc + self.get(i, k) * other.get(j, k)
      }
      out.set(i, j, acc)
    }
  }
  Some(out)
}

///|
/// Transpose.
pub fn Matrix::transpose(self : Matrix) -> Matrix {
  let out = Matrix::zeros(self.cols, self.rows)
  for i = 0; i < self.rows; i = i + 1 {
    for j = 0; j < self.cols; j = j + 1 {
      out.set(j, i, self.get(i, j))
    }
  }
  out
}

///|
/// Sum of every entry.
pub fn Matrix::sum(self : Matrix) -> Double {
  let mut acc = 0.0
  for x in self.data {
    acc = acc + x
  }
  acc
}

///|
/// Mean of every entry.
pub fn Matrix::mean(self : Matrix) -> Double {
  if self.data.length() == 0 {
    return 0.0
  }
  self.sum() / self.data.length().to_double()
}

///|
/// Copy of the backing array, for handing to a parameter slot.
pub fn Matrix::to_array(self : Matrix) -> Array[Double] {
  self.data.copy()
}

///|
/// Rebuild from a flat array produced by `to_array`.
pub fn Matrix::from_flat(
  rows : Int,
  cols : Int,
  data : Array[Double],
) -> Matrix {
  { rows, cols, data, }
}

///|
/// Euclidean norm of the flattened matrix. Used to report gradient
/// magnitudes, which are the most useful single number when
/// diagnosing a training run.
pub fn Matrix::norm(self : Matrix) -> Double {
  let mut acc = 0.0
  for x in self.data {
    acc = acc + x * x
  }
  acc.sqrt()
}

///|
/// Element (Hadamard) product.
pub fn Matrix::hadamard(self : Matrix, other : Matrix) -> Matrix {
  let out = Matrix::zeros(self.rows, self.cols)
  for i = 0; i < self.data.length(); i = i + 1 {
    out.data[i] = self.data[i] * other.data[i]
  }
  out
}

///|
/// Largest absolute entry. A cheap way to catch a diverged run
/// without inspecting every element.
pub fn Matrix::max_abs(self : Matrix) -> Double {
  let mut best = 0.0
  for x in self.data {
    let a = absf(x)
    if a > best {
      best = a
    }
  }
  best
}

///|
/// `true` when every entry is finite. Checked once per training step
/// by the loop in `train.mbt`, because a diverged run is much easier
/// to diagnose at the step where it started than at the step where
/// the loss finally printed `NaN`.
pub fn Matrix::all_finite(self : Matrix) -> Bool {
  for x in self.data {
    if !is_finite(x) {
      return false
    }
  }
  true
}

///|
/// One-line rendering, for the training logs. Truncates long rows so
/// a 64-wide layer does not wrap the terminal.
pub fn Matrix::preview(self : Matrix, max_cols : Int) -> String {
  let cols = if self.cols < max_cols { self.cols } else { max_cols }
  let buf = StringBuilder()
  for i = 0; i < self.rows; i = i + 1 {
    if i > 0 {
      buf.write_string(" ")
    }
    buf.write_char('[')
    for j = 0; j < cols; j = j + 1 {
      if j > 0 {
        buf.write_string(" ")
      }
      buf.write_string(format_double(self.get(i, j)))
    }
    if cols < self.cols {
      buf.write_string(" ...")
    }
    buf.write_char(']')
  }
  buf.to_string()
}

///|
/// Round a double to four decimal places for display. Trailing
/// zeros are left alone: this is a log line, not a serialisation
/// format.
pub fn format_double(x : Double) -> String {
  let scaled = (x * 10000.0).round() / 10000.0
  scaled.to_string()
}
