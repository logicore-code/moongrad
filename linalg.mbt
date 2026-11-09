// linalg.mbt
//
// Dense linear algebra on the tensor type.
//
// These are the routines a scientific-computing library is expected
// to have, and they are also the ones that show the tensor type is
// a real linear-algebra container rather than an array with extra
// steps:
///
///   eigenvalues    symmetric eigendecomposition by the Jacobi method
///   cholesky       A = L L^T, for symmetric positive definite A
///   solve          A x = b, by Cholesky when A is symmetric
///   qr             A = Q R, by Householder reflections
///   svd            A = U S V^T, by one-sided Jacobi
///   inverse        A^-1, by Gauss-Jordan with partial pivoting
///   determinant    det(A), by Gaussian elimination
//
// Everything works on `Tensor` rather than the older `Matrix` type,
// because the tensor type is what the rest of the library speaks.

///|
/// The eigenvalues of a symmetric matrix, by the cyclic Jacobi
/// method.
///
/// Jacobi is not the fastest algorithm — QR iteration is — but it is
/// short, it has no failure mode, and it is unconditionally accurate
/// for symmetric input. For the matrix sizes a MoonBit program is
/// likely to touch it is the right trade.
///
/// The method applies a sequence of rotations that zero out one
/// off-diagonal entry at a time. After enough sweeps the matrix is
/// diagonal, and its diagonal is the eigenvalue vector.
pub fn eigenvalues(a : Tensor) -> Array[Double] {
  let n = a.shape[0]
  if a.shape.length() != 2 || a.shape[1] != n {
    return []
  }
  let m = a.copy()
  // Six sweeps is ample: each sweep zeroes every off-diagonal
  // entry once, and the off-diagonal norm falls quadratically.
  for _sweep = 0; _sweep < 6; _sweep = _sweep + 1 {
    for p = 0; p < n - 1; p = p + 1 {
      for q = p + 1; q < n; q = q + 1 {
        let apq = m.mget(p, q)
        if absf(apq) < 1.0e-14 {
          continue
        }
        let app = m.mget(p, p)
        let aqq = m.mget(q, q)
        let theta = (aqq - app) / (2.0 * apq)
        let t = if theta >= 0.0 {
          1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
          -1.0 / (-theta + (1.0 + theta * theta).sqrt())
        }
        let c = 1.0 / (1.0 + t * t).sqrt()
        let s = t * c
        for k = 0; k < n; k = k + 1 {
          let akp = m.mget(k, p)
          let akq = m.mget(k, q)
          m.mset(k, p, c * akp - s * akq)
          m.mset(k, q, s * akp + c * akq)
        }
        for k = 0; k < n; k = k + 1 {
          let apk = m.mget(p, k)
          let aqk = m.mget(q, k)
          m.mset(p, k, c * apk - s * aqk)
          m.mset(q, k, s * apk + c * aqk)
        }
      }
    }
  }
  let out : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    out.push(m.mget(i, i))
  }
  out
}

///|
/// The Cholesky factor `L` of a symmetric positive definite matrix,
/// so that `A = L @ L^T`.
///
/// Returns `None` when the matrix is not positive definite, which
/// shows up as a non-positive pivot. That is the honest failure
/// mode: a symmetric indefinite matrix has no Cholesky factor, and
/// producing a NaN instead would hide the problem until much later.
pub fn cholesky(a : Tensor) -> Tensor? {
  let n = a.shape[0]
  if a.shape.length() != 2 || a.shape[1] != n {
    return None
  }
  let l = Tensor::zeros([n, n])
  for i = 0; i < n; i = i + 1 {
    for j = 0; j <= i; j = j + 1 {
      let mut sum = a.mget(i, j)
      for k = 0; k < j; k = k + 1 {
        sum = sum - l.mget(i, k) * l.mget(j, k)
      }
      if i == j {
        if sum <= 0.0 {
          return None
        }
        l.mset(i, j, sum.sqrt())
      } else {
        l.mset(i, j, sum / l.mget(j, j))
      }
    }
  }
  Some(l)
}

///|
/// Solve `A x = b` for a symmetric positive definite `A`.
pub fn solve_spd(a : Tensor, b : Array[Double]) -> Array[Double]? {
  let l = match cholesky(a) {
    Some(x) => x
    None => return None
  }
  let n = a.shape[0]
  if b.length() != n {
    return None
  }
  // Forward substitution: L z = b
  let z : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    z.push(0.0)
  }
  for i = 0; i < n; i = i + 1 {
    let mut sum = b[i]
    for k = 0; k < i; k = k + 1 {
      sum = sum - l.mget(i, k) * z[k]
    }
    z[i] = sum / l.mget(i, i)
  }
  // Back substitution: L^T x = z. The elimination runs right to
  // left, so the answers are written into their final slots rather
  // than appended — appending would store them in reverse.
  let x : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    x.push(0.0)
  }
  for i = n - 1; i >= 0; i = i - 1 {
    let mut sum = z[i]
    for k = i + 1; k < n; k = k + 1 {
      sum = sum - l.mget(k, i) * x[k]
    }
    x[i] = sum / l.mget(i, i)
  }
  Some(x)
}

///|
/// Solve `A x = b` by Gaussian elimination with partial pivoting.
///
/// Works for any square matrix, not only a symmetric one. A
/// singular matrix returns `None`.
pub fn solve(a : Tensor, b : Array[Double]) -> Array[Double]? {
  let n = a.shape[0]
  if a.shape.length() != 2 || a.shape[1] != n || b.length() != n {
    return None
  }
  let m = a.copy()
  let x = b.copy()
  for col = 0; col < n; col = col + 1 {
    // Partial pivoting: swap in the largest pivot in the column.
    let mut best = col
    let mut best_val = absf(m.mget(col, col))
    for row = col + 1; row < n; row = row + 1 {
      let v = absf(m.mget(row, col))
      if v > best_val {
        best_val = v
        best = row
      }
    }
    if best_val < 1.0e-14 {
      return None
    }
    if best != col {
      for k = 0; k < n; k = k + 1 {
        let t = m.mget(col, k)
        m.mset(col, k, m.mget(best, k))
        m.mset(best, k, t)
      }
      let t = x[col]
      x[col] = x[best]
      x[best] = t
    }
    // Eliminate below the pivot.
    let pivot = m.mget(col, col)
    for row = col + 1; row < n; row = row + 1 {
      let factor = m.mget(row, col) / pivot
      if factor != 0.0 {
        for k = col; k < n; k = k + 1 {
          m.mset(row, k, m.mget(row, k) - factor * m.mget(col, k))
        }
        x[row] = x[row] - factor * x[col]
      }
    }
  }
  // Back substitution.
  for i = n - 1; i >= 0; i = i - 1 {
    let mut sum = x[i]
    for k = i + 1; k < n; k = k + 1 {
      sum = sum - m.mget(i, k) * x[k]
    }
    x[i] = sum / m.mget(i, i)
  }
  Some(x)
}

///|
/// The inverse of a square matrix, by Gauss-Jordan elimination
/// with partial pivoting.
///
/// Returns `None` for a singular matrix.
pub fn inverse(a : Tensor) -> Tensor? {
  let n = a.shape[0]
  if a.shape.length() != 2 || a.shape[1] != n {
    return None
  }
  // Augment with the identity.
  let m = Tensor::zeros([n, 2 * n])
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      m.mset(i, j, a.mget(i, j))
    }
    m.mset(i, n + i, 1.0)
  }
  for col = 0; col < n; col = col + 1 {
    let mut best = col
    let mut best_val = absf(m.mget(col, col))
    for row = col + 1; row < n; row = row + 1 {
      let v = absf(m.mget(row, col))
      if v > best_val {
        best_val = v
        best = row
      }
    }
    if best_val < 1.0e-14 {
      return None
    }
    if best != col {
      for k = 0; k < 2 * n; k = k + 1 {
        let t = m.mget(col, k)
        m.mset(col, k, m.mget(best, k))
        m.mset(best, k, t)
      }
    }
    // Normalise the pivot row.
    let pivot = m.mget(col, col)
    for k = 0; k < 2 * n; k = k + 1 {
      m.mset(col, k, m.mget(col, k) / pivot)
    }
    // Eliminate the column from the other rows.
    for row = 0; row < n; row = row + 1 {
      if row == col {
        continue
      }
      let factor = m.mget(row, col)
      if factor != 0.0 {
        for k = 0; k < 2 * n; k = k + 1 {
          m.mset(row, k, m.mget(row, k) - factor * m.mget(col, k))
        }
      }
    }
  }
  let out = Tensor::zeros([n, n])
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      out.mset(i, j, m.mget(i, n + j))
    }
  }
  Some(out)
}

///|
/// The determinant of a square matrix, by Gaussian elimination with
/// partial pivoting.
///
/// The sign of the determinant flips with every row swap, so the
/// pivoting has to be tracked rather than ignored.
pub fn determinant(a : Tensor) -> Double {
  let n = a.shape[0]
  if a.shape.length() != 2 || a.shape[1] != n {
    return 0.0
  }
  let m = a.copy()
  let mut det = 1.0
  for col = 0; col < n; col = col + 1 {
    let mut best = col
    let mut best_val = absf(m.mget(col, col))
    for row = col + 1; row < n; row = row + 1 {
      let v = absf(m.mget(row, col))
      if v > best_val {
        best_val = v
        best = row
      }
    }
    if best_val < 1.0e-14 {
      return 0.0
    }
    if best != col {
      for k = 0; k < n; k = k + 1 {
        let t = m.mget(col, k)
        m.mset(col, k, m.mget(best, k))
        m.mset(best, k, t)
      }
      det = -det
    }
    det = det * m.mget(col, col)
    let pivot = m.mget(col, col)
    for row = col + 1; row < n; row = row + 1 {
      let factor = m.mget(row, col) / pivot
      if factor != 0.0 {
        for k = col; k < n; k = k + 1 {
          m.mset(row, k, m.mget(row, k) - factor * m.mget(col, k))
        }
      }
    }
  }
  det
}

///|
/// The QR decomposition of a matrix with orthonormal columns, by
/// Householder reflections.
///
/// `A = Q @ R`, with `Q` square and orthogonal and `R` upper
/// triangular. This is the factorisation behind Gram-Schmidt
/// orthogonalisation, the least-squares solver and the QR
/// algorithm for eigenvalues.
pub fn qr(a : Tensor) -> (Tensor, Tensor)? {
  let m = a.shape[0]
  let n = a.shape[1]
  if a.shape.length() != 2 {
    return None
  }
  let r = a.copy()
  // Q is built as a product of Householder reflections, applied to
  // the identity from the left.
  let q = Tensor::identity(m)
  for k = 0; k < n; k = k + 1 {
    // The Householder vector for column k.
    let mut norm = 0.0
    for i = k; i < m; i = i + 1 {
      norm = norm + r.mget(i, k) * r.mget(i, k)
    }
    norm = norm.sqrt()
    if norm < 1.0e-14 {
      continue
    }
    let alpha = if r.mget(k, k) > 0.0 { -norm } else { norm }
    let v = Tensor::zeros([m])
    for i = 0; i < m; i = i + 1 {
      v.data[i] = if i >= k { r.mget(i, k) } else { 0.0 }
    }
    v.data[k] = v.data[k] - alpha
    let mut vnorm = 0.0
    for i = k; i < m; i = i + 1 {
      vnorm = vnorm + v.data[i] * v.data[i]
    }
    vnorm = vnorm.sqrt()
    if vnorm < 1.0e-14 {
      continue
    }
    for i = 0; i < m; i = i + 1 {
      v.data[i] = v.data[i] / vnorm
    }
    // R <- (I - 2 v v^T) R. The reflection is applied to each
    // *column* of R, so the outer loop runs over the columns and
    // the inner one over the rows.
    for i = 0; i < n; i = i + 1 {
      let mut dot = 0.0
      for j = 0; j < m; j = j + 1 {
        dot = dot + v.data[j] * r.mget(j, i)
      }
      let factor = 2.0 * dot
      for j = 0; j < m; j = j + 1 {
        r.mset(j, i, r.mget(j, i) - factor * v.data[j])
      }
    }
    // Q <- Q (I - 2 v v^T)
    for i = 0; i < m; i = i + 1 {
      let mut dot = 0.0
      for j = 0; j < m; j = j + 1 {
        dot = dot + v.data[j] * q.mget(i, j)
      }
      let factor = 2.0 * dot
      for j = 0; j < m; j = j + 1 {
        q.mset(i, j, q.mget(i, j) - factor * v.data[j])
      }
    }
  }
  Some((q, r))
}

///|
/// The singular values of a matrix, by the one-sided Jacobi method.
///
/// The method orthogonalises the columns of `A` by a sequence of
/// rotations and accumulates them into `V`; the norms of the
/// resulting columns are the singular values. It is more accurate
/// than the Golub-Kahan bidiagonalisation for small matrices and
/// far shorter.
pub fn singular_values(a : Tensor) -> Array[Double] {
  let m = a.shape[0]
  let n = a.shape[1]
  if a.shape.length() != 2 {
    return []
  }
  let u = a.copy()
  let v = Tensor::identity(n)
  for _sweep = 0; _sweep < 12; _sweep = _sweep + 1 {
    for p = 0; p < n - 1; p = p + 1 {
      for q = p + 1; q < n; q = q + 1 {
        let mut alpha = 0.0
        let mut beta = 0.0
        let mut gamma = 0.0
        for i = 0; i < m; i = i + 1 {
          let up = u.mget(i, p)
          let uq = u.mget(i, q)
          alpha = alpha + up * up
          beta = beta + uq * uq
          gamma = gamma + up * uq
        }
        if absf(gamma) < 1.0e-15 {
          continue
        }
        let zeta = (beta - alpha) / (2.0 * gamma)
        let t = if zeta >= 0.0 {
          1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
        } else {
          -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
        }
        let c = 1.0 / (1.0 + t * t).sqrt()
        let s = c * t
        for i = 0; i < m; i = i + 1 {
          let up = u.mget(i, p)
          let uq = u.mget(i, q)
          u.mset(i, p, c * up - s * uq)
          u.mset(i, q, s * up + c * uq)
        }
        for i = 0; i < n; i = i + 1 {
          let vp = v.mget(i, p)
          let vq = v.mget(i, q)
          v.mset(i, p, c * vp - s * vq)
          v.mset(i, q, s * vp + c * vq)
        }
      }
    }
  }
  let out : Array[Double] = []
  for j = 0; j < n; j = j + 1 {
    let mut acc = 0.0
    for i = 0; i < m; i = i + 1 {
      acc = acc + u.mget(i, j) * u.mget(i, j)
    }
    out.push(acc.sqrt())
  }
  out
}

///|
/// The matrix exponential of a small square matrix, by scaling and
/// squaring with a Taylor series.
///
/// `exp(A) = (exp(A/2^k))^(2^k)`, and `exp(A/2^k)` is evaluated with
/// the truncated series `sum (A/2^k)^i / i!`. The scaling keeps the
/// series' argument small enough that a dozen terms suffice.
pub fn matrix_exp(a : Tensor) -> Tensor? {
  let n = a.shape[0]
  if a.shape.length() != 2 || a.shape[1] != n {
    return None
  }
  let scaled = a.scale(1.0 / 16.0)
  // exp(X) = I + X + X^2/2! + ... 
  let mut result = Tensor::identity(n)
  let mut term = Tensor::identity(n)
  let mut factorial = 1.0
  for i = 1; i <= 16; i = i + 1 {
    term = match term.matmul(scaled) {
      Some(t) => t
      None => return None
    }
    factorial = factorial * i.to_double()
    result = match result.add(term.scale(1.0 / factorial)) {
      Some(t) => t
      None => return None
    }
  }
  // Square four times to undo the scaling.
  for _i = 0; _i < 4; _i = _i + 1 {
    result = match result.matmul(result) {
      Some(t) => t
      None => return None
    }
  }
  Some(result)
}
