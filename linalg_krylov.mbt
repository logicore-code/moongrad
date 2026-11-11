// linalg_krylov.mbt
//
// Krylov subspace solvers on the `Matrix` type, and the dense linear
// algebra they lean on.
//
// Conjugate gradients and GMRES are the two workhorses of iterative
// linear algebra. Both build a subspace out of the iterates of `a`
// applied to the residual and then solve a small least-squares
// problem in that subspace. The point of the family is that the only
// thing a solver needs from the matrix is a matrix-vector product,
// so the solvers here work on a matrix that was never assembled.
//
// The file also carries the Jacobi eigendecomposition, the symmetric
// part of a matrix and a spectral norm estimate, because those are
// what a caller needs in order to decide whether a matrix is worth
// handing to a Krylov solver at all.
//
// The routines work on `Matrix` rather than on `Tensor`, because the
// solvers are host-side: they run to completion and return a plain
// array of doubles, and none of them belongs in a computation graph.

///|
/// The matrix-vector product `a * x`.
///
/// Returns `None` when the number of columns of `a` and the length of
/// `x` disagree. A mismatch means a vector and a matrix have been
/// mixed up, and padding or truncating to hide it would turn a small
/// mistake into a wrong answer.
pub fn matvec(a : Matrix, x : Array[Double]) -> Array[Double]? {
  let rows = a.nrows()
  let cols = a.ncols()
  if cols != x.length() {
    return None
  }
  let out : Array[Double] = []
  for i = 0; i < rows; i = i + 1 {
    let mut acc = 0.0
    for j = 0; j < cols; j = j + 1 {
      acc = acc + a.get(i, j) * x[j]
    }
    out.push(acc)
  }
  Some(out)
}

///|
/// The inner product of two vectors.
fn vec_dot(a : Array[Double], b : Array[Double]) -> Double {
  let n = if a.length() < b.length() { a.length() } else { b.length() }
  let mut acc = 0.0
  for i = 0; i < n; i = i + 1 {
    acc = acc + a[i] * b[i]
  }
  acc
}

///|
/// The Euclidean norm of a vector.
fn vec_norm(a : Array[Double]) -> Double {
  let mut acc = 0.0
  for x in a {
    acc = acc + x * x
  }
  acc.sqrt()
}

///|
/// The entries of one column of a matrix.
fn column_of(m : Matrix, j : Int32) -> Array[Double] {
  let out : Array[Double] = []
  for i = 0; i < m.nrows(); i = i + 1 {
    out.push(m.get(i, j))
  }
  out
}

///|
/// Solve `a x = b` for a square system by Gaussian elimination with
/// partial pivoting.
///
/// A singular matrix returns `None`. The pivot threshold is relative
/// to the size of the matrix, so a system whose entries are all tiny
/// is not mistaken for a singular one.
fn gauss_solve(a : Matrix, b : Array[Double]) -> Array[Double]? {
  let n = a.nrows()
  if a.ncols() != n || b.length() != n {
    return None
  }
  let m = Matrix::zeros(n, n)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      m.set(i, j, a.get(i, j))
    }
  }
  let x = b.copy()
  // The scale of the matrix, for the singularity test.
  let mut scale = 0.0
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let magnitude = absf(m.get(i, j))
      if magnitude > scale {
        scale = magnitude
      }
    }
  }
  if scale == 0.0 {
    return None
  }
  for col = 0; col < n; col = col + 1 {
    // Partial pivoting: swap in the largest pivot in the column.
    let mut best = col
    let mut best_val = absf(m.get(col, col))
    for row = col + 1; row < n; row = row + 1 {
      let magnitude = absf(m.get(row, col))
      if magnitude > best_val {
        best_val = magnitude
        best = row
      }
    }
    if best_val < 1.0e-14 * scale {
      return None
    }
    if best != col {
      for k = 0; k < n; k = k + 1 {
        let t = m.get(col, k)
        m.set(col, k, m.get(best, k))
        m.set(best, k, t)
      }
      let t = x[col]
      x[col] = x[best]
      x[best] = t
    }
    // Eliminate below the pivot.
    let pivot = m.get(col, col)
    for row = col + 1; row < n; row = row + 1 {
      let factor = m.get(row, col) / pivot
      if factor != 0.0 {
        for k = col; k < n; k = k + 1 {
          m.set(row, k, m.get(row, k) - factor * m.get(col, k))
        }
        x[row] = x[row] - factor * x[col]
      }
    }
  }
  // Back substitution.
  for i = n - 1; i >= 0; i = i - 1 {
    let mut sum = x[i]
    for k = i + 1; k < n; k = k + 1 {
      sum = sum - m.get(i, k) * x[k]
    }
    x[i] = sum / m.get(i, i)
  }
  Some(x)
}

///|
/// Solve `a x = b` for a symmetric positive definite `a` by conjugate
/// gradients.
///
/// The iteration is the textbook one: `r0 = b - a x0`, `p0 = r0`,
/// `alpha = (r'r) / (p' a p)`, `x += alpha p`, `r -= alpha a p`,
/// `beta = (r'r_new) / (r'r_old)`, `p = r_new + beta p`.
///
/// Two curvatures are checked rather than assumed. The curvature
/// `r' a r` of the residual direction has to be positive, and so does
/// the curvature `p' a p` of every search direction; a non-positive
/// one means the matrix is not positive definite and the iteration
/// has no meaning. The matrix is checked once up front as well, so
/// that the refusal does not depend on which right-hand side
/// happened to arrive.
///
/// The residual is recomputed as `b - a x` on every iteration instead
/// of being carried along recursively. The recursive residual drifts
/// away from the true one, and the drift is what makes an iteration
/// look converged when it is not.
///
/// Returns `None` when `a` is not square, when the dimensions
/// disagree, or when `a` is not positive definite.
pub fn conj_grad(
  a : Matrix,
  b : Array[Double],
  x0 : Array[Double],
  max_iter : Int32,
  tol : Double,
) -> Array[Double]? {
  let n = a.nrows()
  if a.nrows() != a.ncols() || b.length() != n || x0.length() != n {
    return None
  }
  // The matrix-level check: some direction of the symmetric part of
  // `a` has negative curvature, so there is no conjugate gradient
  // iteration for this matrix, whatever the right-hand side is.
  if has_negative_curvature(a) {
    return None
  }
  let ax0 = match matvec(a, x0) {
    Some(v) => v
    None => return None
  }
  let mut r : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    r.push(b[i] - ax0[i])
  }
  let mut rsq = vec_dot(r, r)
  if rsq <= 0.0 {
    // The initial guess already solves the system.
    return Some(x0.copy())
  }
  // The curvature of the residual direction. A non-positive one means
  // the iteration is heading into a region where the matrix curves
  // the wrong way, and the whole recurrence breaks down there.
  let ar = match matvec(a, r) {
    Some(v) => v
    None => return None
  }
  if vec_dot(r, ar) <= 0.0 {
    return None
  }
  let x = x0.copy()
  let p = r.copy()
  for _iter = 0; _iter < max_iter; _iter = _iter + 1 {
    let ap = match matvec(a, p) {
      Some(v) => v
      None => return None
    }
    let pap = vec_dot(p, ap)
    if pap <= 0.0 {
      // Breakdown: the search direction has no positive curvature, so
      // the matrix is not positive definite.
      return None
    }
    let alpha = rsq / pap
    for i = 0; i < n; i = i + 1 {
      x[i] = x[i] + alpha * p[i]
      r[i] = r[i] - alpha * ap[i]
    }
    // Recompute the residual from scratch; the recursive one drifts.
    let ax = match matvec(a, x) {
      Some(v) => v
      None => return None
    }
    let r2 : Array[Double] = []
    for i = 0; i < n; i = i + 1 {
      r2.push(b[i] - ax[i])
    }
    let rsq_new = vec_dot(r2, r2)
    if rsq_new.sqrt() < tol {
      return Some(x)
    }
    let beta = rsq_new / rsq
    for i = 0; i < n; i = i + 1 {
      p[i] = r2[i] + beta * p[i]
    }
    r = r2
    rsq = rsq_new
  }
  Some(x)
}

///|
/// Solve `a x = b` for a general square matrix by restarted GMRES.
///
/// The Krylov space is built with the Arnoldi process, orthogonalised
/// by modified Gram-Schmidt, and the iterate is chosen by solving the
/// least-squares problem `min || h y - beta e_1 ||` in that space.
/// The normal equations of that problem are solved by Gaussian
/// elimination with partial pivoting, so the only dense linear algebra
/// is on a matrix of the size of the Krylov space.
///
/// The space is restarted every `min(max_iter, 20)` steps. Twenty
/// vectors is the usual compromise between the work spent per
/// restart and the memory the basis needs; a longer space converges
/// further per restart but costs more to keep.
///
/// The matrix does not have to be symmetric, and it does not have to
/// be positive definite. It does have to be square and it does have
/// to be invertible, or the iteration stalls and the best iterate
/// found so far is returned.
///
/// Returns `None` when the dimensions disagree.
pub fn gmres(
  a : Matrix,
  b : Array[Double],
  x0 : Array[Double],
  max_iter : Int32,
  tol : Double,
) -> Array[Double]? {
  let n = a.nrows()
  if a.nrows() != a.ncols() || b.length() != n || x0.length() != n {
    return None
  }
  let x = x0.copy()
  if n == 0 || max_iter <= 0 {
    return Some(x)
  }
  let window = if max_iter < 20 { max_iter } else { 20 }
  let mut used = 0
  while used < max_iter {
    // The residual of the current iterate.
    let ax = match matvec(a, x) {
      Some(v) => v
      None => return None
    }
    let residual : Array[Double] = []
    for i = 0; i < n; i = i + 1 {
      residual.push(b[i] - ax[i])
    }
    let beta = vec_norm(residual)
    if beta < tol {
      return Some(x)
    }
    let steps = if max_iter - used < window { max_iter - used } else { window }
    // The Krylov basis v_0 ... v_steps and the Hessenberg matrix h.
    let v = Matrix::zeros(n, steps + 1)
    for i = 0; i < n; i = i + 1 {
      v.set(i, 0, residual[i] / beta)
    }
    let h = Matrix::zeros(steps + 1, steps)
    let mut built = 0
    for j = 0; j < steps; j = j + 1 {
      let w = match matvec(a, column_of(v, j)) {
        Some(value) => value
        None => return None
      }
      let wnorm = vec_norm(w)
      // Modified Gram-Schmidt: the projections onto the basis vectors
      // are subtracted one at a time, which keeps the basis
      // orthogonal even when the matrix is badly conditioned.
      for i = 0; i <= j; i = i + 1 {
        let vi = column_of(v, i)
        let hij = vec_dot(w, vi)
        h.set(i, j, hij)
        for t = 0; t < n; t = t + 1 {
          w[t] = w[t] - hij * vi[t]
        }
      }
      let hnext = vec_norm(w)
      h.set(j + 1, j, hnext)
      built = j + 1
      if hnext <= 1.0e-13 * wnorm {
        // The Krylov space is closed: the next vector would be a
        // combination of the ones already there, and the solution
        // inside the space is exact. The test is relative to the size
        // of `a v_j`, because the rounding noise in the
        // orthogonalisation is of that order.
        break
      }
      for t = 0; t < n; t = t + 1 {
        v.set(t, j + 1, w[t] / hnext)
      }
    }
    // The least-squares problem in the Krylov space.
    let coefficients = match least_squares(h, built, beta) {
      Some(y) => y
      None =>
        // The space is degenerate; the current iterate is the best
        // one available.
        return Some(x)
    }
    // x <- x + v y
    for j = 0; j < built; j = j + 1 {
      let vj = column_of(v, j)
      for i = 0; i < n; i = i + 1 {
        x[i] = x[i] + coefficients[j] * vj[i]
      }
    }
    used = used + built
  }
  Some(x)
}

///|
/// The coefficients that minimise `|| h y - beta e_1 ||`, where the
/// first `m` columns of `h` span the Krylov space.
///
/// The normal equations of the least-squares problem are solved by
/// Gaussian elimination with partial pivoting. Forming them squares
/// the condition number of `h`, which is the price of solving the
/// problem this way rather than by a rotation of the basis.
fn least_squares(h : Matrix, m : Int32, beta : Double) -> Array[Double]? {
  if m <= 0 {
    return None
  }
  let rows = h.nrows()
  let normal = Matrix::zeros(m, m)
  let rhs : Array[Double] = []
  for _i = 0; _i < m; _i = _i + 1 {
    rhs.push(0.0)
  }
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      let mut acc = 0.0
      for k = 0; k < rows; k = k + 1 {
        acc = acc + h.get(k, i) * h.get(k, j)
      }
      normal.set(i, j, acc)
    }
    // The right-hand side is beta e_1, so only the first row of `h`
    // contributes to it.
    rhs[i] = h.get(0, i) * beta
  }
  gauss_solve(normal, rhs)
}

///|
/// Solve `a x = b` for a symmetric positive definite `a` by conjugate
/// gradients, starting from zero.
///
/// The iteration budget is four times the dimension and the tolerance
/// is tight, which is enough for the method to reach the accuracy the
/// direct solvers reach on a well-conditioned system.
pub fn cg_solve_spd(a : Matrix, b : Array[Double]) -> Array[Double]? {
  let n = a.nrows()
  let x0 : Array[Double] = []
  for _i = 0; _i < n; _i = _i + 1 {
    x0.push(0.0)
  }
  conj_grad(a, b, x0, 4 * n, 1.0e-10)
}

///|
/// The eigenvalues and eigenvectors of a symmetric matrix, by the
/// Jacobi method.
///
/// The matrix is assumed symmetric. The largest off-diagonal entry is
/// picked out and a plane rotation is applied to annihilate it; the
/// rotations are accumulated into an orthogonal matrix `v`, so that
///
///     a = v diag(lambda) v^T
///
/// The eigenvalues come back in ascending order, with the matching
/// eigenvectors as the columns of `v` in the same order.
///
/// `sweeps` is the rotation budget, counted in full sweeps of the
/// matrix: one sweep is `n (n - 1) / 2` rotations. Choosing the
/// largest pivot each time converges faster than the cyclic order,
/// so the same budget buys more accuracy here than it does in a
/// cyclic sweep.
///
/// Returns `None` when the matrix is not square.
pub fn symmetric_eigen_jacobi_all(
  a_in : Matrix,
  sweeps : Int32,
) -> (Array[Double], Matrix)? {
  let n = a_in.nrows()
  if a_in.nrows() != a_in.ncols() {
    return None
  }
  let a = Matrix::zeros(n, n)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      a.set(i, j, a_in.get(i, j))
    }
  }
  // The size of the matrix, for the convergence test.
  let mut scale = 0.0
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      scale = scale + a.get(i, j) * a.get(i, j)
    }
  }
  scale = scale.sqrt()
  if scale == 0.0 {
    scale = 1.0
  }
  let v = Matrix::identity(n)
  let budget = sweeps * (n * (n - 1) / 2)
  let mut rotations = 0
  while rotations < budget {
    // The largest off-diagonal entry is the pivot.
    let mut p = 0
    let mut q = 0
    let mut largest = 0.0
    for i = 0; i < n - 1; i = i + 1 {
      for j = i + 1; j < n; j = j + 1 {
        let magnitude = absf(a.get(i, j))
        if magnitude > largest {
          largest = magnitude
          p = i
          q = j
        }
      }
    }
    if largest <= 1.0e-14 * scale {
      // The matrix is diagonal to within the rounding noise.
      break
    }
    let apq = a.get(p, q)
    let app = a.get(p, p)
    let aqq = a.get(q, q)
    let theta = (aqq - app) / (2.0 * apq)
    let t = if theta >= 0.0 {
      1.0 / (theta + (1.0 + theta * theta).sqrt())
    } else {
      -1.0 / (-theta + (1.0 + theta * theta).sqrt())
    }
    let c = 1.0 / (1.0 + t * t).sqrt()
    let s = t * c
    // a <- j^T a j: the columns first, then the rows.
    for k = 0; k < n; k = k + 1 {
      let akp = a.get(k, p)
      let akq = a.get(k, q)
      a.set(k, p, c * akp - s * akq)
      a.set(k, q, s * akp + c * akq)
    }
    for k = 0; k < n; k = k + 1 {
      let apk = a.get(p, k)
      let aqk = a.get(q, k)
      a.set(p, k, c * apk - s * aqk)
      a.set(q, k, s * apk + c * aqk)
    }
    // v <- v j: the eigenvectors follow the matrix.
    for k = 0; k < n; k = k + 1 {
      let vkp = v.get(k, p)
      let vkq = v.get(k, q)
      v.set(k, p, c * vkp - s * vkq)
      v.set(k, q, s * vkp + c * vkq)
    }
    rotations = rotations + 1
  }
  // The eigenvalues are the diagonal of the rotated matrix. Sort them
  // into ascending order, carrying the eigenvectors along.
  let values : Array[Double] = []
  let order : Array[Int32] = []
  for i = 0; i < n; i = i + 1 {
    values.push(a.get(i, i))
    order.push(i)
  }
  for i = 0; i < n - 1; i = i + 1 {
    let mut best = i
    for j = i + 1; j < n; j = j + 1 {
      if values[j] < values[best] {
        best = j
      }
    }
    if best != i {
      let t = values[i]
      values[i] = values[best]
      values[best] = t
      let u = order[i]
      order[i] = order[best]
      order[best] = u
    }
  }
  let eigenvalues = values
  let vectors = Matrix::zeros(n, n)
  for j = 0; j < n; j = j + 1 {
    let source = order[j]
    let mut norm = 0.0
    for i = 0; i < n; i = i + 1 {
      norm = norm + v.get(i, source) * v.get(i, source)
    }
    norm = norm.sqrt()
    for i = 0; i < n; i = i + 1 {
      let value = v.get(i, source)
      vectors.set(i, j, if norm > 0.0 { value / norm } else { value })
    }
  }
  Some((eigenvalues, vectors))
}

///|
/// The symmetric part of a matrix, `(a + a^T) / 2`.
///
/// A matrix put together by two separate computations can pick up
/// asymmetry of the order of the floating-point noise, and the noise
/// is not harmless: it moves the eigenvalues that a Newton step or a
/// conjugate gradient iteration depends on.
///
/// Only the square leading block of a non-square matrix is
/// symmetrised, since a non-square matrix has no transpose to add.
pub fn symmetric_part(a : Matrix) -> Matrix {
  let rows = a.nrows()
  let cols = a.ncols()
  let block = if rows < cols { rows } else { cols }
  let out = Matrix::zeros(rows, cols)
  for i = 0; i < block; i = i + 1 {
    for j = 0; j < block; j = j + 1 {
      out.set(i, j, 0.5 * (a.get(i, j) + a.get(j, i)))
    }
  }
  out
}

///|
/// Whether the symmetric part of a matrix has a direction of negative
/// curvature.
///
/// Only the symmetric part matters: the curvature of a matrix in a
/// direction `x` is `x' a x`, and the skew-symmetric part of `a`
/// contributes nothing to it. A negative eigenvalue means some
/// direction curves the wrong way, so the matrix is indefinite and no
/// conjugate gradient iteration exists for it.
///
/// The test is made against the size of the matrix, so an eigenvalue
/// that is only negative because of rounding is not mistaken for a
/// direction of genuinely negative curvature.
fn has_negative_curvature(a : Matrix) -> Bool {
  let n = a.nrows()
  if n == 0 || a.nrows() != a.ncols() {
    return false
  }
  let (eigenvalues, _vectors) = match
    symmetric_eigen_jacobi_all(symmetric_part(a), 30) {
    Some(pair) => pair
    None => return false
  }
  let mut scale = 0.0
  for e in eigenvalues {
    let magnitude = absf(e)
    if magnitude > scale {
      scale = magnitude
    }
  }
  eigenvalues[0] < -1.0e-12 * scale
}

///|
/// A bound on the spectral norm of a matrix, by the power method on
/// `a^T a`.
///
/// The largest eigenvalue of `a^T a` is the square of the spectral
/// norm, so the power method on it converges to the answer. A hundred
/// iterations is more than enough: the iteration converges
/// geometrically at the rate of the ratio of the two largest
/// eigenvalues, and the estimate is good long before the budget runs
/// out.
///
/// Returns `0.0` for an empty matrix.
pub fn spectral_norm_bound(a : Matrix) -> Double {
  let rows = a.nrows()
  let cols = a.ncols()
  if rows == 0 || cols == 0 {
    return 0.0
  }
  // b = a^T a, whose largest eigenvalue is the square of the spectral
  // norm.
  let at = a.transpose()
  let b = match at.mul(a) {
    Some(m) => m
    None => return 0.0
  }
  let n = b.nrows()
  // The starting direction has distinct entries, so that it is not
  // accidentally orthogonal to the dominant eigenvector. It is scaled
  // to unit length, because the Rayleigh quotient below assumes a
  // unit vector.
  let v : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    v.push(1.0 / (i + 1).to_double())
  }
  let start = vec_norm(v)
  for i = 0; i < n; i = i + 1 {
    v[i] = v[i] / start
  }
  let mut lambda = 0.0
  let mut best = 0.0
  for _iter = 0; _iter < 100; _iter = _iter + 1 {
    let bv = match matvec(b, v) {
      Some(value) => value
      None => return 0.0
    }
    // The Rayleigh quotient of the current direction.
    let quotient = vec_dot(v, bv)
    if quotient > best {
      best = quotient
    }
    let norm = vec_norm(bv)
    if norm == 0.0 {
      // The matrix annihilates the starting direction; there is
      // nothing left to iterate on.
      break
    }
    for i = 0; i < n; i = i + 1 {
      v[i] = bv[i] / norm
    }
    if absf(quotient - lambda) <= 1.0e-12 * absf(quotient) {
      lambda = quotient
      break
    }
    lambda = quotient
  }
  best.sqrt()
}
