// linalg_test.mbt
//
// The dense linear algebra routines.
//
// Each test checks a property that has to hold regardless of how
// the algorithm is implemented: an eigenvector is an eigenvector, a
// Cholesky factor reconstructs the original, a QR decomposition
// reconstructs it too, and a solve really solves the system. Those
// are the statements a reviewer can verify by hand, and they catch
// the class of bug where the code runs but is subtly wrong.

///|
/// The eigenvalues of a square matrix, or a failure the test reports.
fn eigenvalues_of(m : @moongrad.Tensor) -> Array[Double] raise {
  match @moongrad.eigenvalues(m) {
    Some(x) => x
    None => {
      check(false, "the matrix should have eigenvalues")
      []
    }
  }
}

///|
/// The singular values of a matrix, or a failure the test reports.
fn singular_values_of(m : @moongrad.Tensor) -> Array[Double] raise {
  match @moongrad.singular_values(m) {
    Some(x) => x
    None => {
      check(false, "the matrix should have singular values")
      []
    }
  }
}

///|
/// The determinant of a square matrix, or a failure the test reports.
fn determinant_of(m : @moongrad.Tensor) -> Double raise {
  match @moongrad.determinant(m) {
    Some(x) => x
    None => {
      check(false, "the matrix should have a determinant")
      0.0
    }
  }
}

///|
fn mat(rows : Array[Array[Double]]) -> @moongrad.Tensor raise {
  match @moongrad.Tensor::matrix(rows) {
    Some(t) => t
    None => {
      check(false, "the matrix should build")
      @moongrad.Tensor::zeros([0, 0])
    }
  }
}

///|
fn close(a : Double, b : Double, tol : Double) -> Bool {
  a - b < tol && b - a < tol
}

// ---------------------------------------------------------------------------
// Eigenvalues
// ---------------------------------------------------------------------------

///|
test "eigenvalues_of_a_diagonal_matrix_are_its_diagonal" {
  let m = mat([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
  let ev = eigenvalues_of(m)
  assert_eq(ev.length(), 3)
  let mut total = 0.0
  for e in ev {
    total = total + e
  }
  // The eigenvalues of a diagonal matrix are its diagonal entries,
  // in some order, so the sum is the trace.
  check(close(total, 6.0, 1.0e-9), "the trace should be 6, got \{total}")
}

///|
test "eigenvalues_of_a_symmetric_matrix_sum_to_the_trace" {
  let m = mat([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
  let ev = eigenvalues_of(m)
  let mut total = 0.0
  for e in ev {
    total = total + e
  }
  check(
    close(total, 9.0, 1.0e-9),
    "the @moongrad.eigenvalues of a symmetric matrix sum to its trace, got \{total}",
  )
}

///|
test "eigenvalues_of_a_rotation_like_matrix" {
  // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
  let m = mat([[2.0, 1.0], [1.0, 2.0]])
  let ev = eigenvalues_of(m)
  let mut lo = ev[0]
  let mut hi = ev[1]
  if lo > hi {
    let t = lo
    lo = hi
    hi = t
  }
  check(close(lo, 1.0, 1.0e-9), "the smaller eigenvalue should be 1, got \{lo}")
  check(close(hi, 3.0, 1.0e-9), "the larger eigenvalue should be 3, got \{hi}")
}

///|
test "eigenvalues_of_a_rank_one_matrix_are_n_and_zero" {
  // [[2, 2], [2, 2]] has eigenvalues 4 and 0.
  let m = mat([[2.0, 2.0], [2.0, 2.0]])
  let ev = eigenvalues_of(m)
  let mut lo = ev[0]
  let mut hi = ev[1]
  if lo > hi {
    let t = lo
    lo = hi
    hi = t
  }
  check(close(lo, 0.0, 1.0e-9), "the smaller eigenvalue should be 0, got \{lo}")
  check(close(hi, 4.0, 1.0e-9), "the larger eigenvalue should be 4, got \{hi}")
}

// ---------------------------------------------------------------------------
// Cholesky and the SPD solver
// ---------------------------------------------------------------------------

///|
test "cholesky_reconstructs_the_original_matrix" {
  let m = mat([[4.0, 2.0, 2.0], [2.0, 5.0, 1.0], [2.0, 1.0, 6.0]])
  let l = match @moongrad.cholesky(m) {
    Some(x) => x
    None => {
      check(false, "the matrix is positive definite")
      return
    }
  }
  // A = L @ L^T
  let lt = match l.transpose([1, 0]) {
    Some(x) => x
    None => return
  }
  let product = match l.matmul(lt) {
    Some(x) => x
    None => {
      check(false, "the product should exist")
      return
    }
  }
  for i = 0; i < 3; i = i + 1 {
    for j = 0; j < 3; j = j + 1 {
      check(
        close(product.mget(i, j), m.mget(i, j), 1.0e-9),
        "L @ L^T should equal A at (\{i}, \{j})",
      )
    }
  }
}

///|
test "cholesky_rejects_an_indefinite_matrix" {
  let m = mat([[1.0, 2.0], [2.0, 1.0]])
  let l = @moongrad.cholesky(m)
  let rejected = match l {
    None => true
    Some(_) => false
  }
  check(rejected, "an indefinite matrix has no Cholesky factor")
}

///|
test "solve_spd_finds_the_right_answer" {
  // Solve [[4, 2], [2, 5]] x = [12, 17].
  let m = mat([[4.0, 2.0], [2.0, 5.0]])
  let x = match @moongrad.solve_spd(m, [12.0, 17.0]) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  // The exact solution is [1.625, 2.75]: 4x + 2y = 12 and
  // 2x + 5y = 17 give 4y = 11.
  check(close(x[0], 1.625, 1.0e-9), "x0 should be 1.625, got \{x[0]}")
  check(close(x[1], 2.75, 1.0e-9), "x1 should be 2.75, got \{x[1]}")
}

///|
test "solve_handles_a_general_matrix" {
  // Solve [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]] x = [8, -11, -3].
  let m = mat([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
  let x = match @moongrad.solve(m, [8.0, -11.0, -3.0]) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  // The exact solution is [2, 3, -1].
  check(close(x[0], 2.0, 1.0e-9), "x0 should be 2, got \{x[0]}")
  check(close(x[1], 3.0, 1.0e-9), "x1 should be 3, got \{x[1]}")
}

///|
test "solve_swaps_the_rows_when_the_pivot_is_zero" {
  // The first pivot of [[0, 1], [1, 0]] is zero, so the only way
  // forward is to swap the rows. A solver that divides by the pivot
  // without looking for a larger one returns a number that is not a
  // solution.
  let a = match @moongrad.Tensor::matrix([[0.0, 1.0], [1.0, 0.0]]) {
    Some(t) => t
    None => return
  }
  let x = match @moongrad.solve(a, [1.0, 2.0]) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  // The exact solution is [2, 1].
  check(close(x[0], 2.0, 1.0e-9), "x0 should be 2, got \{x[0]}")
  check(close(x[1], 1.0, 1.0e-9), "x1 should be 1, got \{x[1]}")
}

///|
test "solve_rejects_a_singular_system" {
  let m = mat([[1.0, 2.0], [2.0, 4.0]])
  let x = @moongrad.solve(m, [1.0, 2.0])
  let rejected = match x {
    None => true
    Some(_) => false
  }
  check(rejected, "a singular system has no unique solution")
}

// ---------------------------------------------------------------------------
// Inverse and determinant
// ---------------------------------------------------------------------------

///|
test "inverse_multiplies_back_to_the_identity" {
  let m = mat([[4.0, 7.0], [2.0, 6.0]])
  let inv = match @moongrad.inverse(m) {
    Some(x) => x
    None => {
      check(false, "the matrix should be invertible")
      return
    }
  }
  let product = match m.matmul(inv) {
    Some(x) => x
    None => return
  }
  for i = 0; i < 2; i = i + 1 {
    for j = 0; j < 2; j = j + 1 {
      let want = if i == j { 1.0 } else { 0.0 }
      check(
        close(product.mget(i, j), want, 1.0e-9),
        "A @ A^-1 should be the identity at (\{i}, \{j})",
      )
    }
  }
}

///|
test "inverse_of_a_singular_matrix_is_refused" {
  let m = mat([[1.0, 2.0], [2.0, 4.0]])
  let inv = @moongrad.inverse(m)
  let rejected = match inv {
    None => true
    Some(_) => false
  }
  check(rejected, "a singular matrix has no @moongrad.inverse")
}

///|
test "determinant_of_a_known_matrix" {
  // The determinant of [[1, 2], [3, 4]] is -2.
  let m = mat([[1.0, 2.0], [3.0, 4.0]])
  check(
    close(determinant_of(m), -2.0, 1.0e-9),
    "the @moongrad.determinant should be -2, got \{determinant_of(m)}",
  )
}

///|
test "determinant_of_a_triangular_matrix_is_the_product_of_the_diagonal" {
  let m = mat([[2.0, 3.0, 4.0], [0.0, 5.0, 6.0], [0.0, 0.0, 7.0]])
  check(
    close(determinant_of(m), 70.0, 1.0e-9),
    "the @moongrad.determinant should be 70, got \{determinant_of(m)}",
  )
}

///|
test "determinant_of_a_singular_matrix_is_zero" {
  let m = mat([[1.0, 2.0], [2.0, 4.0]])
  check(
    close(determinant_of(m), 0.0, 1.0e-9),
    "a singular matrix has @moongrad.determinant zero",
  )
}

// ---------------------------------------------------------------------------
// QR and singular values
// ---------------------------------------------------------------------------

///|
test "qr_reconstructs_the_original_matrix" {
  let m = mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  let (q, r) = match @moongrad.qr(m) {
    Some(x) => x
    None => {
      check(false, "the decomposition should exist")
      return
    }
  }
  let product = match q.matmul(r) {
    Some(x) => x
    None => return
  }
  for i = 0; i < 3; i = i + 1 {
    for j = 0; j < 2; j = j + 1 {
      check(
        close(product.mget(i, j), m.mget(i, j), 1.0e-9),
        "Q @ R should equal A at (\{i}, \{j})",
      )
    }
  }
}

///|
test "qr_produces_orthonormal_columns" {
  let m = mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  let (q, _r) = match @moongrad.qr(m) {
    Some(x) => x
    None => return
  }
  // Q^T Q should be the identity.
  let qt = match q.transpose([1, 0]) {
    Some(x) => x
    None => return
  }
  let product = match qt.matmul(q) {
    Some(x) => x
    None => return
  }
  for i = 0; i < 2; i = i + 1 {
    for j = 0; j < 2; j = j + 1 {
      let want = if i == j { 1.0 } else { 0.0 }
      check(
        close(product.mget(i, j), want, 1.0e-9),
        "Q^T Q should be the identity at (\{i}, \{j})",
      )
    }
  }
}

///|
test "singular_values_of_a_diagonal_matrix_are_its_diagonal" {
  let m = mat([[3.0, 0.0], [0.0, 2.0]])
  let sv = singular_values_of(m)
  let mut lo = sv[0]
  let mut hi = sv[1]
  if lo > hi {
    let t = lo
    lo = hi
    hi = t
  }
  check(close(lo, 2.0, 1.0e-9), "the smaller singular value should be 2")
  check(close(hi, 3.0, 1.0e-9), "the larger singular value should be 3")
}

///|
test "singular_values_of_a_rank_one_matrix_include_a_zero" {
  // [[1, 2], [2, 4]] has rank one, so one singular value is zero.
  let m = mat([[1.0, 2.0], [2.0, 4.0]])
  let sv = singular_values_of(m)
  let mut smallest = sv[0]
  for v in sv {
    if v < smallest {
      smallest = v
    }
  }
  check(
    close(smallest, 0.0, 1.0e-8),
    "a rank-one matrix has a zero singular value, got \{smallest}",
  )
}

// ---------------------------------------------------------------------------
// Matrix exponential
// ---------------------------------------------------------------------------

///|
test "matrix_exp_of_a_zero_matrix_is_the_identity" {
  let m = @moongrad.Tensor::zeros([3, 3])
  let e = match @moongrad.matrix_exp(m) {
    Some(x) => x
    None => {
      check(false, "the exponential should exist")
      return
    }
  }
  for i = 0; i < 3; i = i + 1 {
    for j = 0; j < 3; j = j + 1 {
      let want = if i == j { 1.0 } else { 0.0 }
      check(
        close(e.mget(i, j), want, 1.0e-9),
        "exp(0) is the identity at (\{i}, \{j})",
      )
    }
  }
}

///|
test "matrix_exp_of_a_nilpotent_matrix_is_the_identity_plus_the_matrix" {
  // A matrix whose square is zero has exp(A) = I + A exactly.
  // [[0, 1, 0], [0, 0, 0], [0, 0, 0]] squares to zero.
  let m = mat([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
  let e = match @moongrad.matrix_exp(m) {
    Some(x) => x
    None => {
      check(false, "the exponential should exist")
      return
    }
  }
  check(close(e.mget(0, 0), 1.0, 1.0e-9), "the diagonal stays one")
  check(
    close(e.mget(0, 1), 1.0, 1.0e-9),
    "the first superdiagonal is unchanged",
  )
  check(close(e.mget(0, 2), 0.0, 1.0e-9), "the second superdiagonal stays zero")
  check(close(e.mget(1, 1), 1.0, 1.0e-9), "the diagonal stays one")
}

///|
test "matrix_exp_includes_the_quadratic_term" {
  // [[0, 1, 2], [0, 0, 3], [0, 0, 0]] has A^2[0][2] = 3 and A^3 =
  // 0, so exp(A) = I + A + A^2/2 and the second superdiagonal picks
  // up 3/2 on top of A's 2.
  let m = mat([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
  let e = match @moongrad.matrix_exp(m) {
    Some(x) => x
    None => {
      check(false, "the exponential should exist")
      return
    }
  }
  check(
    close(e.mget(0, 2), 3.5, 1.0e-9),
    "the second superdiagonal should be 2 + 3/2 = 3.5, got \{e.mget(0, 2)}",
  )
  check(
    close(e.mget(1, 2), 3.0, 1.0e-9),
    "the first superdiagonal is unchanged",
  )
}

///|
test "matrix_exp_satisfies_the_power_series_to_second_order" {
  // For a small matrix, exp(A) = I + A + A^2/2 to within the
  // truncation error of the series, so the quadratic term has to be
  // right.
  let m = mat([[0.0, 1.0], [0.0, 0.0]])
  let e = match @moongrad.matrix_exp(m) {
    Some(x) => x
    None => return
  }
  // exp([[0,1],[0,0]]) = [[1,1],[0,1]] exactly, because the matrix
  // is nilpotent of index two.
  check(close(e.mget(0, 1), 1.0, 1.0e-9), "the off-diagonal is one")
  check(close(e.mget(1, 1), 1.0, 1.0e-9), "the diagonal is one")
}

///|
test "the_matrix_exponential_of_a_matrix_that_is_not_nilpotent" {
  // The other exponential tests use a matrix whose square is zero, and
  // for such a matrix every truncation of the series is exact — so
  // they cannot tell a series that stops early from one that does not.
  // This one does: for [[0, a], [a, 0]] the exponential is
  // `cosh(a) I + sinh(a) [[0, 1], [1, 0]]`, and the off-diagonal entry
  // is `sinh(a)`, which the truncated series does not reach.
  let m = mat([[0.0, 4.0], [4.0, 0.0]])
  let e = match @moongrad.matrix_exp(m) {
    Some(t) => t
    None => {
      check(false, "the exponential should exist")
      return
    }
  }
  // sinh(4) = 27.289917197127752.
  check(
    close(e.mget(0, 1), 27.289917197127752, 1.0e-6),
    "the off-diagonal entry should be sinh(4): \{e.mget(0, 1)}",
  )
  // cosh(4) = 27.308232836016487.
  check(
    close(e.mget(0, 0), 27.308232836016487, 1.0e-6),
    "the diagonal entry should be cosh(4): \{e.mget(0, 0)}",
  )
}

///|
test "the_eigenvalues_of_a_symmetric_matrix_are_the_exact_ones" {
  // The eigenvalues of [[2, 1, 0], [1, 2, 1], [0, 1, 2]] are
  // 2 - sqrt(2), 2 and 2 + sqrt(2). The decomposition sweeps a fixed
  // number of times and stops, and a sweep budget that is too small
  // leaves the off-diagonal entries standing: the answer is then
  // close to the right one but not equal to it, which is the kind of
  // wrong that a tolerance chosen to match the error would hide.
  let m = mat([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
  let values = match @moongrad.eigenvalues(m) {
    Some(v) => v
    None => {
      check(false, "the matrix should have eigenvalues")
      return
    }
  }
  let sorted = values.copy()
  for i = 0; i < sorted.length(); i = i + 1 {
    for j = i + 1; j < sorted.length(); j = j + 1 {
      if sorted[j] < sorted[i] {
        let t = sorted[i]
        sorted[i] = sorted[j]
        sorted[j] = t
      }
    }
  }
  let root = 2.0.sqrt()
  check(
    close(sorted[0], 2.0 - root, 1.0e-12),
    "the smallest eigenvalue should be 2 - sqrt(2): \{sorted[0]}",
  )
  check(
    close(sorted[1], 2.0, 1.0e-12),
    "the middle eigenvalue should be 2: \{sorted[1]}",
  )
  check(
    close(sorted[2], 2.0 + root, 1.0e-12),
    "the largest eigenvalue should be 2 + sqrt(2): \{sorted[2]}",
  )
}

///|
test "the_r_factor_of_a_qr_decomposition_has_a_negative_diagonal" {
  // The sign of the diagonal of R is a convention, and a convention
  // that is not written down is one that can change. This one is:
  // every diagonal entry of R is negative, and every one of them is
  // non-zero.
  let m = mat([[3.0, 1.0], [4.0, 2.0]])
  let (q, r) = match @moongrad.qr(m) {
    Some(t) => t
    None => {
      check(false, "the decomposition should exist")
      return
    }
  }
  let n = r.shape[0]
  for i = 0; i < n; i = i + 1 {
    check(r.mget(i, i) < 0.0, "the diagonal of R should be negative")
    check(r.mget(i, i) != 0.0, "the diagonal of R should not be zero")
  }
  // The decomposition still has to reconstruct the matrix.
  let qr = match q.matmul(r) {
    Some(t) => t
    None => {
      check(false, "the product should exist")
      return
    }
  }
  for i = 0; i < 2; i = i + 1 {
    for j = 0; j < 2; j = j + 1 {
      check(
        close(qr.mget(i, j), m.mget(i, j), 1.0e-9),
        "Q R should be the matrix again",
      )
    }
  }
}
