// linalg_krylov.mbt
//
// Krylov subspace solvers on the `Matrix` type, and the dense linear
// algebra they lean on.
//
// Conjugate gradients and GMRES are the two workhorses of iterative
// linear algebra. Both build a subspace out of the iterates of `a`
// applied to the residual and then solve a small least-squares
// problem in that subspace. The point of the family is that the
// iteration needs nothing from the matrix but a matrix-vector
// product, so the solvers here work on a matrix that was never
// assembled.
//
// Conjugate gradients is the one exception, and it is a small one.
// The recurrence is only defined for a symmetric positive definite
// matrix, so the solver checks those two properties before it
// starts. The check is a Cholesky factorisation, which is cubic in
// the dimension and needs no eigendecomposition; it happens once per
// solve, and the iteration itself is still nothing but matrix-vector
// products.
//
// The file also carries the Jacobi eigendecomposition, the symmetric
// part of a matrix and bounds on the spectral norm, because those are
// what a caller needs in order to decide whether a matrix is worth
// handing to a Krylov solver at all.
//
// The routines work on `Matrix` rather than on `Tensor`, because the
// solvers are host-side: they run to completion and return a plain
// array of doubles, and none of them belongs in a computation graph.
//
// The rule the solvers follow is that a vector which is not a
// solution is not an answer. A routine that cannot finish says so:
// `None` means the solve did not happen, and a routine that reports
// what it achieved says whether it converged.

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
/// The matrix-vector product of a matrix and a vector whose length
/// agrees with the width of the matrix.
///
/// The dimensions have been checked by the caller, so the product
/// cannot fail and is not wrapped in an option.
fn matvec_vector(a : Matrix, x : Array[Double]) -> Array[Double] {
  let out : Array[Double] = []
  for i = 0; i < a.nrows(); i = i + 1 {
    let mut acc = 0.0
    for j = 0; j < a.ncols(); j = j + 1 {
      acc = acc + a.get(i, j) * x[j]
    }
    out.push(acc)
  }
  out
}

///|
/// The residual `b - a x` of a candidate solution.
fn residual_vector(
  a : Matrix,
  x : Array[Double],
  b : Array[Double],
) -> Array[Double] {
  let out : Array[Double] = []
  for i = 0; i < a.nrows(); i = i + 1 {
    let mut acc = 0.0
    for j = 0; j < a.ncols(); j = j + 1 {
      acc = acc + a.get(i, j) * x[j]
    }
    out.push(b[i] - acc)
  }
  out
}

///|
/// Whether a matrix agrees with its transpose to within a relative
/// tolerance.
///
/// The tolerance is relative to the size of the matrix, so a matrix
/// whose entries are all tiny is not mistaken for an asymmetric one,
/// and a matrix that picked up an asymmetry of the order of the
/// floating-point noise is not either.
fn is_symmetric(a : Matrix, tol : Double) -> Bool {
  let rows = a.nrows()
  let cols = a.ncols()
  if rows != cols {
    return false
  }
  let mut scale = 0.0
  for i = 0; i < rows; i = i + 1 {
    for j = 0; j < cols; j = j + 1 {
      let magnitude = absf(a.get(i, j))
      if magnitude > scale {
        scale = magnitude
      }
    }
  }
  let limit = tol * scale
  for i = 0; i < rows; i = i + 1 {
    for j = i + 1; j < cols; j = j + 1 {
      if absf(a.get(i, j) - a.get(j, i)) > limit {
        return false
      }
    }
  }
  true
}

///|
/// Whether the symmetric part of a square matrix is positive definite.
///
/// The test is a Cholesky factorisation: a symmetric matrix is
/// positive definite exactly when it has one, `a = l l^T`, and the
/// factorisation is found in a single pass over the matrix, in time
/// cubic in the dimension. A matrix with an eigenvalue that is
/// negative somewhere has no such factorisation, so the test refuses
/// it.
///
/// The test is the strict one. A pivot that comes out non-positive is
/// a refusal, even for a matrix that is positive definite with an
/// eigenvalue that is positive only to within the rounding noise.
/// Such a matrix has no conjugate gradient iteration to run anyway:
/// the recurrence divides by that curvature.
fn is_positive_definite(a : Matrix) -> Bool {
  let n = a.nrows()
  if a.nrows() != a.ncols() {
    return false
  }
  if n == 0 {
    // The empty matrix is positive definite: there is no direction in
    // it that curves the wrong way.
    return true
  }
  let s = symmetric_part(a)
  // The Cholesky factor, filled in as it goes.
  let l = Matrix::zeros(n, n)
  for j = 0; j < n; j = j + 1 {
    let mut diagonal = s.get(j, j)
    for k = 0; k < j; k = k + 1 {
      let v = l.get(j, k)
      diagonal = diagonal - v * v
    }
    if !(diagonal > 0.0) {
      // Not positive definite. The comparison is written the other way
      // round on purpose: it also catches a pivot that is not a number.
      return false
    }
    let d = diagonal.sqrt()
    l.set(j, j, d)
    for i = j + 1; i < n; i = i + 1 {
      let mut acc = s.get(i, j)
      for k = 0; k < j; k = k + 1 {
        acc = acc - l.get(i, k) * l.get(j, k)
      }
      l.set(i, j, acc / d)
    }
  }
  true
}

///|
/// Solve `a x = b` for a symmetric positive definite `a` by conjugate
/// gradients.
///
/// The iteration is the textbook one: `r0 = b - a x0`, `p0 = r0`,
/// `alpha = (r'r) / (p' a p)`, `x += alpha p`, `r -= alpha a p`,
/// `beta = (r'r_new) / (r'r_old)`, `p = r_new + beta p`.
///
/// The matrix has to be symmetric. The recurrence is run on `a`
/// itself and not on its symmetric part, and conjugate gradients is
/// the method that minimises the residual over the Krylov space of
/// `a`: the step lengths and the conjugacy of the search directions
/// are what make that minimisation work, and both of them need a
/// symmetric matrix. A matrix that is not symmetric has no conjugate
/// gradient iteration, and the recurrence runs to the end without
/// arriving at a solution of anything. A matrix that agrees with its
/// transpose to within a relative tolerance of `1e-12` counts as
/// symmetric, and the recurrence is run on that matrix as it stands.
///
/// The matrix also has to be positive definite. That is checked once
/// up front, with a Cholesky factorisation of the symmetric part of
/// `a`, so that the refusal does not depend on which right-hand side
/// happened to arrive. The check is cubic in the dimension and needs
/// no eigendecomposition.
///
/// The residual is recomputed as `b - a x` on every iteration instead
/// of being carried along recursively. The recursive residual drifts
/// away from the true one, and the drift is what makes an iteration
/// look converged when it is not.
///
/// Returns `None` when `a` is not square, when the dimensions
/// disagree, when `a` is not symmetric, when `a` is not positive
/// definite, and when the iteration budget ran out before the
/// residual reached `tol`. The last of those matters as much as the
/// others: an iterate that is not a solution is not an answer, so it
/// is reported as a failure instead of being handed back as if it had
/// succeeded.
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
  if !is_symmetric(a, 1.0e-12) {
    // The recurrence below runs on `a` itself, so a matrix that is
    // not symmetric has no conjugate gradient iteration.
    return None
  }
  if !is_positive_definite(a) {
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
  if rsq.sqrt() <= tol {
    // The initial guess already solves the system to the tolerance.
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
    // The comparison is not strict: a tolerance of zero means an
    // exact solution, and a residual that is exactly zero satisfies
    // it.
    if rsq_new.sqrt() <= tol {
      return Some(x)
    }
    let beta = rsq_new / rsq
    for i = 0; i < n; i = i + 1 {
      p[i] = r2[i] + beta * p[i]
    }
    r = r2
    rsq = rsq_new
  }
  // The budget ran out before the residual reached the tolerance. The
  // iterate is not a solution, so it is not returned.
  None
}

///|
/// Why a Krylov iteration stopped.
///
/// These are the three ways an iteration can end: it found the
/// answer, it ran out of budget, or it stopped making progress.
pub(all) enum KrylovStop {
  /// The residual reached the tolerance.
  Converged
  /// The iteration budget ran out before the residual reached the
  /// tolerance.
  IterationLimit
  /// The iteration stopped making progress: the least-squares problem
  /// in the Krylov space was singular.
  Stalled
}

///|
/// What a Krylov solve achieved.
///
/// The iterate in the report is the best one of the run, meaning the
/// one with the smallest residual, so that a caller can look at what
/// the iteration achieved even when it did not converge.
pub(all) struct KrylovReport {
  /// The iterate with the smallest residual of the run.
  x : Array[Double]
  /// The norm of the residual of that iterate.
  residual : Double
  /// How many Arnoldi steps the iteration spent.
  iterations : Int32
  /// Why the iteration stopped.
  stop : KrylovStop
}

///|
/// Solve `a x = b` for a general square matrix by restarted GMRES.
///
/// The Krylov space is built with the Arnoldi process, orthogonalised
/// by modified Gram-Schmidt, and the iterate is chosen by solving the
/// least-squares problem `min || h y - beta e_1 ||` in that space.
/// The least-squares problem is solved by a QR factorisation with
/// Givens rotations, so the only dense linear algebra is on a matrix
/// of the size of the Krylov space.
///
/// The space is restarted every `min(max_iter, 20, n)` steps at most.
/// Twenty vectors is the usual compromise between the work spent per
/// restart and the memory the basis needs; a longer space converges
/// further per restart but costs more to keep. The space is never
/// longer than the matrix is wide, because there is no room in it for
/// more than `n` independent vectors.
///
/// The matrix does not have to be symmetric, and it does not have to
/// be positive definite. It does have to be square, and it does have
/// to be invertible for the iteration to finish; a matrix that is not
/// makes the least-squares problem in the Krylov space singular, and
/// the iteration stalls.
///
/// Returns `None` when the dimensions disagree, and `None` when the
/// iteration did not reach the tolerance. A vector that is not a
/// solution is not an answer, so a run that ran out of budget or
/// stalled is reported as a failure rather than handed back as if it
/// had succeeded. `gmres_report` is the variant that says what the
/// iteration achieved instead.
pub fn gmres(
  a : Matrix,
  b : Array[Double],
  x0 : Array[Double],
  max_iter : Int32,
  tol : Double,
) -> Array[Double]? {
  match gmres_report(a, b, x0, max_iter, tol) {
    Some(report) =>
      match report.stop {
        Converged => Some(report.x)
        IterationLimit => None
        Stalled => None
      }
    None => None
  }
}

///|
/// Solve `a x = b` for a general square matrix by restarted GMRES,
/// and report what the iteration achieved.
///
/// The iteration is the one `gmres` runs; see there for the details.
///
/// The report holds the best iterate of the run, meaning the one with
/// the smallest residual, and the reason the iteration stopped. A run
/// that ran out of budget or that stalled is reported as such, with
/// the best iterate it found, so that a caller can see how far the
/// iteration got instead of being handed a vector that is not a
/// solution.
///
/// Returns `None` when the dimensions disagree.
pub fn gmres_report(
  a : Matrix,
  b : Array[Double],
  x0 : Array[Double],
  max_iter : Int32,
  tol : Double,
) -> KrylovReport? {
  let n = a.nrows()
  if a.nrows() != a.ncols() || b.length() != n || x0.length() != n {
    return None
  }
  let x = x0.copy()
  // The best iterate of the run, and the residual that goes with it.
  // A restart that makes the residual worse does not throw away the
  // progress of the restarts before it.
  let mut best_x = x.copy()
  let mut best_residual = vec_norm(residual_vector(a, x, b))
  let mut used = 0
  if best_residual <= tol {
    // The initial guess already solves the system to the tolerance.
    return Some({
      x: best_x,
      residual: best_residual,
      iterations: 0,
      stop: Converged,
    })
  }
  let mut stop = IterationLimit
  while used < max_iter {
    // The residual of the current iterate.
    let residual = residual_vector(a, x, b)
    let beta = vec_norm(residual)
    if beta <= tol {
      stop = Converged
      break
    }
    if beta < best_residual {
      best_residual = beta
      best_x = x.copy()
    }
    // The length of the Krylov space: never more than the width of
    // the matrix, never more than the budget that is left.
    let mut steps = max_iter - used
    if steps > 20 {
      steps = 20
    }
    if steps > n {
      steps = n
    }
    let v0 = residual.copy()
    for i = 0; i < n; i = i + 1 {
      v0[i] = v0[i] / beta
    }
    let (v, h, built, _closed) = arnoldi_basis(a, v0, steps)
    used = used + built
    let coefficients = match least_squares(h, built, beta) {
      Some(y) => y
      None => {
        // The least-squares problem in the Krylov space is rank
        // deficient: the space does not carry the information the
        // iteration needs and there is no progress left to make. The
        // best iterate of the run is kept and the stall is reported,
        // rather than the iterate from before the restart being
        // handed back as if it were a solution.
        stop = Stalled
        break
      }
    }
    // x <- x + v y
    for j = 0; j < built; j = j + 1 {
      let vj = column_of(v, j)
      for i = 0; i < n; i = i + 1 {
        x[i] = x[i] + coefficients[j] * vj[i]
      }
    }
    // The residual of the new iterate. It is checked here and not only
    // at the start of the next restart, so that a run which converges
    // on the last step of the budget is still recognised as one.
    let new_residual = vec_norm(residual_vector(a, x, b))
    if new_residual < best_residual {
      best_residual = new_residual
      best_x = x.copy()
    }
    if new_residual <= tol {
      stop = Converged
      break
    }
  }
  Some({ x: best_x, residual: best_residual, iterations: used, stop, })
}

///|
/// The Arnoldi basis of the Krylov space of `a` that starts at `v0`.
///
/// The basis is the columns of the first matrix and the Hessenberg
/// matrix of the process is the second one; the number of vectors that
/// were built is the third. The fourth says whether the space closed
/// before the budget ran out.
///
/// The orthogonalisation is done twice over the same basis vectors.
/// The first pass leaves a residue along them, because they are not
/// exactly orthogonal to each other, and that residue is what hides
/// the closing of the space: with a single pass the test for the
/// breakdown misses it, the iteration goes on to build one more vector
/// in a space that has no room left in it, and the least-squares
/// problem it then solves is rank deficient.
fn arnoldi_basis(
  a : Matrix,
  v0 : Array[Double],
  steps : Int32,
) -> (Matrix, Matrix, Int32, Bool) {
  let n = a.nrows()
  let v = Matrix::zeros(n, steps + 1)
  for i = 0; i < n; i = i + 1 {
    v.set(i, 0, v0[i])
  }
  let h = Matrix::zeros(steps + 1, steps)
  let mut built = 0
  let mut closed = false
  for j = 0; j < steps; j = j + 1 {
    let w = matvec_vector(a, column_of(v, j))
    let wnorm = vec_norm(w)
    // Modified Gram-Schmidt: the projections onto the basis vectors
    // are subtracted one at a time, which keeps the basis orthogonal
    // even when the matrix is badly conditioned.
    for i = 0; i <= j; i = i + 1 {
      let vi = column_of(v, i)
      let hij = vec_dot(w, vi)
      h.set(i, j, h.get(i, j) + hij)
      for t = 0; t < n; t = t + 1 {
        w[t] = w[t] - hij * vi[t]
      }
    }
    // The second pass over the same projections takes away what the
    // first one left along the basis, which is the rounding of the
    // orthogonalisation and not a real component of the vector.
    for i = 0; i <= j; i = i + 1 {
      let vi = column_of(v, i)
      let correction = vec_dot(w, vi)
      h.set(i, j, h.get(i, j) + correction)
      for t = 0; t < n; t = t + 1 {
        w[t] = w[t] - correction * vi[t]
      }
    }
    let hnext = vec_norm(w)
    h.set(j + 1, j, hnext)
    built = j + 1
    if hnext <= 1.0e-12 * wnorm {
      // The Krylov space is closed: the next vector would be a
      // combination of the ones already there, and the solution
      // inside the space is exact. The test is relative to the size
      // of `a v_j`, because the rounding noise in the
      // orthogonalisation is of that order.
      closed = true
      break
    }
    for t = 0; t < n; t = t + 1 {
      v.set(t, j + 1, w[t] / hnext)
    }
  }
  (v, h, built, closed)
}

///|
/// The coefficients that minimise `|| h y - beta e_1 ||`, where the
/// first `m` columns of `h` span the Krylov space.
///
/// The problem is solved by a QR factorisation with Givens rotations,
/// which are applied to the right-hand side along with the matrix.
/// Forming the normal equations instead would square the condition
/// number of `h`, which is the price of solving the problem that way;
/// the rotations cost a square root each and keep the condition
/// number of the problem itself.
///
/// Returns `None` when the least-squares problem is rank deficient,
/// which means the Krylov space did not carry the information the
/// iteration needs.
fn least_squares(h : Matrix, m : Int32, beta : Double) -> Array[Double]? {
  if m <= 0 {
    return None
  }
  if h.nrows() < m + 1 {
    return None
  }
  // A working copy of the first `m` columns of `h`, row-major, with
  // one row more than there are columns: the extra row is where the
  // part of the right-hand side that the space cannot reach ends up.
  let a : Array[Double] = []
  for _i = 0; _i < (m + 1) * m; _i = _i + 1 {
    a.push(0.0)
  }
  for j = 0; j < m; j = j + 1 {
    for i = 0; i <= m; i = i + 1 {
      a[i * m + j] = h.get(i, j)
    }
  }
  // The right-hand side `beta e_1`, to be rotated along with the
  // matrix.
  let g : Array[Double] = []
  for _i = 0; _i <= m; _i = _i + 1 {
    g.push(0.0)
  }
  g[0] = beta
  for j = 0; j < m; j = j + 1 {
    for i = m; i > j; i = i - 1 {
      // A rotation of the rows `i - 1` and `i` that zeroes the entry
      // of column `j` below the diagonal.
      let upper = a[(i - 1) * m + j]
      let lower = a[i * m + j]
      let r = (upper * upper + lower * lower).sqrt()
      if r == 0.0 {
        continue
      }
      let c = upper / r
      let s = lower / r
      for k = j; k < m; k = k + 1 {
        let u = a[(i - 1) * m + k]
        let l = a[i * m + k]
        a[(i - 1) * m + k] = c * u + s * l
        a[i * m + k] = -s * u + c * l
      }
      let gu = g[i - 1]
      let gl = g[i]
      g[i - 1] = c * gu + s * gl
      g[i] = -s * gu + c * gl
    }
  }
  // The diagonal of the triangular factor, for the test of rank.
  let mut scale = 0.0
  for i = 0; i < m; i = i + 1 {
    let v = absf(a[i * m + i])
    if v > scale {
      scale = v
    }
  }
  if scale == 0.0 {
    // The Krylov space carries nothing at all.
    return None
  }
  // Back substitution.
  let y : Array[Double] = []
  for _i = 0; _i < m; _i = _i + 1 {
    y.push(0.0)
  }
  for i = m - 1; i >= 0; i = i - 1 {
    let mut acc = g[i]
    for k = i + 1; k < m; k = k + 1 {
      acc = acc - a[i * m + k] * y[k]
    }
    let pivot = a[i * m + i]
    if absf(pivot) <= 1.0e-14 * scale {
      // The least-squares problem is rank deficient: the Krylov space
      // does not span the directions that the right-hand side needs.
      return None
    }
    y[i] = acc / pivot
  }
  Some(y)
}

///|
/// Solve `a x = b` for a symmetric positive definite `a` by conjugate
/// gradients, starting from zero.
///
/// The iteration budget is four times the dimension and the tolerance
/// is tight, which is enough for the method to reach the accuracy the
/// direct solvers reach on a well-conditioned system.
///
/// Returns `None` when the matrix is not square, not symmetric or not
/// positive definite, and `None` when the iteration did not reach the
/// tolerance. See `conj_grad` for the details.
pub fn cg_solve_spd(a : Matrix, b : Array[Double]) -> Array[Double]? {
  let n = a.nrows()
  let x0 : Array[Double] = []
  for _i = 0; _i < n; _i = _i + 1 {
    x0.push(0.0)
  }
  conj_grad(a, b, x0, 4 * n, 1.0e-10)
}

///|
/// What a Jacobi eigendecomposition achieved.
pub(all) struct JacobiEigen {
  /// The eigenvalues, in ascending order.
  eigenvalues : Array[Double]
  /// The matching eigenvectors, as the columns of the matrix.
  eigenvectors : Matrix
  /// The norm of the off-diagonal part of the rotated matrix: how far
  /// the iteration got. The eigenvalues are wrong by at most about
  /// this much.
  residual : Double
  /// Whether the iteration got far enough for the result to be a
  /// decomposition of the matrix.
  converged : Bool
}

///|
/// The eigenvalues and eigenvectors of a symmetric matrix, by the
/// Jacobi method, together with the error the iteration achieved.
///
/// The matrix has to be symmetric: the plane rotations below are the
/// rotations of a symmetric matrix, and the result is a decomposition
/// of the matrix that was given, not of some matrix near it. A
/// matrix that agrees with its transpose to within a relative
/// tolerance of `1e-12` counts as symmetric.
///
/// The eigenvalues come back in ascending order, with the matching
/// eigenvectors as the columns of `v` in the same order.
///
/// Returns `None` only when the matrix is not square or not
/// symmetric. An iteration that ran out of budget is reported
/// through `converged` and `residual` rather than through a missing
/// result, so that a caller can see how close the iteration came.
pub fn symmetric_eigen_jacobi_report(
  a_in : Matrix,
  sweeps : Int32,
) -> JacobiEigen? {
  let n = a_in.nrows()
  if a_in.nrows() != a_in.ncols() {
    return None
  }
  if !is_symmetric(a_in, 1.0e-12) {
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
  // The off-diagonal part of the rotated matrix: how far the
  // iteration got. The eigenvalues of the rotated matrix differ from
  // the eigenvalues of the original one by at most this much, so a
  // result that is handed back without this test is a result that
  // might not be an eigenvalue at all.
  let mut residual = 0.0
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      if i != j {
        let v = a.get(i, j)
        residual = residual + v * v
      }
    }
  }
  residual = residual.sqrt()
  // The sum of the squares overflows for a matrix with very large
  // entries, and an infinite residual compares as "at most" an
  // infinite scale, so the iteration would report a decomposition it
  // never reached. A non-finite measure is a failure, not a success.
  let converged = is_finite(residual) &&
    is_finite(scale) &&
    residual <= 1.0e-12 * scale
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
  Some({ eigenvalues, eigenvectors: vectors, residual, converged, })
}

///|
/// The eigenvalues and eigenvectors of a symmetric matrix, by the
/// Jacobi method.
///
/// The matrix has to be symmetric: the plane rotations below are the
/// rotations of a symmetric matrix, and the result is a decomposition
/// of the matrix that was given, not of some matrix near it. A matrix
/// that agrees with its transpose to within a relative tolerance of
/// `1e-12` counts as symmetric. The largest off-diagonal entry is
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
/// The iteration has to finish before the result means anything: the
/// eigenvalues of a matrix whose off-diagonal part has not been
/// annihilated are not the eigenvalues of that matrix. The routine
/// therefore measures how far the iteration got and refuses to hand
/// back a decomposition that has not converged.
/// `symmetric_eigen_jacobi_report` is the variant that reports the
/// error that was achieved, for the caller that wants to see how
/// close the iteration came.
///
/// Returns `None` when the matrix is not square, when it is not
/// symmetric to within a relative tolerance, and when the iteration
/// did not converge.
pub fn symmetric_eigen_jacobi_all(
  a_in : Matrix,
  sweeps : Int32,
) -> (Array[Double], Matrix)? {
  match symmetric_eigen_jacobi_report(a_in, sweeps) {
    Some(report) =>
      if report.converged {
        Some((report.eigenvalues, report.eigenvectors))
      } else {
        None
      }
    None => None
  }
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
/// A lower bound on the spectral norm of a matrix, by the power method
/// on `a^T a`.
///
/// The largest eigenvalue of `a^T a` is the square of the spectral
/// norm, and the power method converges to it from below: the
/// Rayleigh quotient it computes is never larger than the largest
/// eigenvalue. The square root of that quotient is therefore never
/// larger than the spectral norm either, which makes this a lower
/// bound and not an upper one. A hundred iterations is more than
/// enough for the estimate to be close: the iteration converges
/// geometrically at the rate of the ratio of the two largest
/// eigenvalues, and the estimate is good long before the budget runs
/// out.
///
/// The upper bound on the spectral norm is
/// `spectral_norm_upper_bound`.
///
/// Returns `0.0` for an empty matrix.
pub fn spectral_norm_lower_bound(a : Matrix) -> Double {
  let rows = a.nrows()
  let cols = a.ncols()
  if rows == 0 || cols == 0 {
    return 0.0
  }
  // b = a^T a, whose largest eigenvalue is the square of the spectral
  // norm. The matrix is scaled first: the product overflows for a
  // matrix with very large entries, and an infinite bound is not a
  // bound at all.
  let mut largest = 0.0
  for i = 0; i < a.nrows(); i = i + 1 {
    for j = 0; j < a.ncols(); j = j + 1 {
      let v = a.get(i, j)
      let m = if v < 0.0 { -v } else { v }
      if m > largest {
        largest = m
      }
    }
  }
  if largest == 0.0 {
    return 0.0
  }
  let scaled = Matrix::zeros(a.nrows(), a.ncols())
  for i = 0; i < a.nrows(); i = i + 1 {
    for j = 0; j < a.ncols(); j = j + 1 {
      scaled.set(i, j, a.get(i, j) / largest)
    }
  }
  let at = scaled.transpose()
  let b = match at.mul(scaled) {
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
  // The iteration ran on the scaled matrix, so the estimate has to
  // be scaled back.
  largest * best.sqrt()
}

///|
/// An upper bound on the spectral norm of a matrix.
///
/// The bound is `sqrt(||a||_1 ||a||_inf)`: the square root of the
/// product of the largest column sum and the largest row sum of the
/// matrix. The spectral norm is at most that product, and the product
/// is at most `n` times the square of the spectral norm, so the bound
/// is loose by at most a factor of the dimension.
///
/// Where `spectral_norm_lower_bound` is the estimate of the power
/// method, which is never larger than the truth, this one is never
/// smaller than it.
///
/// Returns `0.0` for an empty matrix.
pub fn spectral_norm_upper_bound(a : Matrix) -> Double {
  let rows = a.nrows()
  let cols = a.ncols()
  if rows == 0 || cols == 0 {
    return 0.0
  }
  let column_sums : Array[Double] = []
  for _j = 0; _j < cols; _j = _j + 1 {
    column_sums.push(0.0)
  }
  let mut row_sum = 0.0
  let mut column_sum = 0.0
  for i = 0; i < rows; i = i + 1 {
    let mut row = 0.0
    for j = 0; j < cols; j = j + 1 {
      let magnitude = absf(a.get(i, j))
      row = row + magnitude
      column_sums[j] = column_sums[j] + magnitude
    }
    if row > row_sum {
      row_sum = row
    }
  }
  for s in column_sums {
    if s > column_sum {
      column_sum = s
    }
  }
  (row_sum * column_sum).sqrt()
}
