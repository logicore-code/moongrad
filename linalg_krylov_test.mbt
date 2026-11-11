// linalg_krylov_test.mbt
//
// The Krylov solvers and the dense linear algebra around them.
//
// The properties checked here are the ones a caller depends on: a
// solver returns the solution of the system it was given, it refuses
// a system it cannot solve instead of returning a wrong number, and
// an eigendecomposition reconstructs the matrix it came from. Those
// are the statements a reviewer can verify by hand, and they catch
// the class of bug where the code runs but is subtly wrong.

///|
/// A matrix built from rows of numbers.
fn mkmat(rows : Array[Array[Double]]) -> @moongrad.Matrix {
  let n = rows.length()
  let cols = if n == 0 { 0 } else { rows[0].length() }
  let m = @moongrad.Matrix::zeros(n, cols)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < cols; j = j + 1 {
      m.set(i, j, rows[i][j])
    }
  }
  m
}

///|
/// Whether two numbers are within `tol` of each other.
fn near(a : Double, b : Double, tol : Double) -> Bool {
  a - b < tol && b - a < tol
}

///|
/// The residual of `a x = b`.
fn residual_of(
  a : @moongrad.Matrix,
  x : Array[Double],
  b : Array[Double],
) -> Double {
  let mut acc = 0.0
  for i = 0; i < b.length(); i = i + 1 {
    let mut ax = 0.0
    for j = 0; j < a.ncols(); j = j + 1 {
      ax = ax + a.get(i, j) * x[j]
    }
    let r = b[i] - ax
    acc = acc + r * r
  }
  acc.sqrt()
}

// ---------------------------------------------------------------------------
// Matrix-vector product
// ---------------------------------------------------------------------------

///|
test "matvec_multiplies_a_matrix_by_a_vector" {
  let a = mkmat([[1.0, 2.0], [3.0, 4.0]])
  let x = match @moongrad.matvec(a, [1.0, 1.0]) {
    Some(v) => v
    None => {
      check(false, "the product should exist")
      return
    }
  }
  check(x.length() == 2, "the product should have two entries")
  check(near(x[0], 3.0, 1.0e-12), "the first entry should be 3, got \{x[0]}")
  check(near(x[1], 7.0, 1.0e-12), "the second entry should be 7, got \{x[1]}")
}

///|
test "matvec_refuses_a_vector_of_the_wrong_length" {
  let a = mkmat([[1.0, 2.0], [3.0, 4.0]])
  let x = @moongrad.matvec(a, [1.0, 2.0, 3.0])
  let refused = match x {
    None => true
    Some(_) => false
  }
  check(refused, "a vector of the wrong length has no matrix product")
}

// ---------------------------------------------------------------------------
// Conjugate gradients
// ---------------------------------------------------------------------------

///|
test "conj_grad_solves_a_two_by_two_system" {
  // Solve [[4, 1], [1, 3]] x = [1, 2]. The exact solution is
  // [1/11, 7/11].
  let a = mkmat([[4.0, 1.0], [1.0, 3.0]])
  let x = match @moongrad.conj_grad(a, [1.0, 2.0], [0.0, 0.0], 10, 1.0e-12) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  check(near(x[0], 1.0 / 11.0, 1.0e-9), "x0 should be 1/11, got \{x[0]}")
  check(near(x[1], 7.0 / 11.0, 1.0e-9), "x1 should be 7/11, got \{x[1]}")
}

///|
test "conj_grad_solves_a_five_by_five_system" {
  // A symmetric, diagonally dominant matrix, with the right-hand
  // side built from a known solution so that the answer can be
  // checked rather than assumed.
  let a = mkmat([
    [4.0, -1.0, 0.0, 0.0, 0.0],
    [-1.0, 4.0, -1.0, 0.0, 0.0],
    [0.0, -1.0, 4.0, -1.0, 0.0],
    [0.0, 0.0, -1.0, 4.0, -1.0],
    [0.0, 0.0, 0.0, -1.0, 4.0],
  ])
  let want = [1.0, 2.0, 3.0, 4.0, 5.0]
  let b = [2.0, 4.0, 6.0, 8.0, 16.0]
  let x = match
    @moongrad.conj_grad(a, b, [0.0, 0.0, 0.0, 0.0, 0.0], 40, 1.0e-12) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  for i = 0; i < 5; i = i + 1 {
    check(
      near(x[i], want[i], 1.0e-8),
      "x\{i} should be \{want[i]}, got \{x[i]}",
    )
  }
  check(
    residual_of(a, x, b) < 1.0e-8,
    "the residual should be small, got \{residual_of(a, x, b)}",
  )
}

///|
test "conj_grad_refuses_an_indefinite_matrix" {
  // [[1, 2], [2, 1]] has eigenvalues 3 and -1, so it curves the
  // wrong way along the direction of the residual and there is no
  // conjugate gradient iteration for it.
  let a = mkmat([[1.0, 2.0], [2.0, 1.0]])
  let x = @moongrad.conj_grad(a, [1.0, -1.0], [0.0, 0.0], 10, 1.0e-10)
  let refused = match x {
    None => true
    Some(_) => false
  }
  check(refused, "an indefinite matrix has no conjugate gradient solve")
}

///|
test "conj_grad_refuses_an_indefinite_matrix_whatever_the_right_hand_side" {
  // The same matrix, with a right-hand side whose curvature happens
  // to be positive. The refusal must not depend on the residual.
  let a = mkmat([[1.0, 2.0], [2.0, 1.0]])
  let x = @moongrad.conj_grad(a, [1.0, 1.0], [0.0, 0.0], 10, 1.0e-10)
  let refused = match x {
    None => true
    Some(_) => false
  }
  check(refused, "an indefinite matrix has no conjugate gradient solve")
}

///|
test "conj_grad_refuses_a_dimension_mismatch" {
  let a = mkmat([[4.0, 1.0], [1.0, 3.0]])
  let x = @moongrad.conj_grad(a, [1.0, 2.0, 3.0], [0.0, 0.0], 10, 1.0e-10)
  let refused = match x {
    None => true
    Some(_) => false
  }
  check(refused, "a right-hand side of the wrong length has no solution")
}

///|
test "conj_grad_refuses_a_non_square_matrix" {
  let a = mkmat([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0]])
  let x = @moongrad.conj_grad(a, [1.0, 2.0], [0.0, 0.0], 10, 1.0e-10)
  let refused = match x {
    None => true
    Some(_) => false
  }
  check(refused, "a non-square matrix has no conjugate gradient solve")
}

// ---------------------------------------------------------------------------
// GMRES
// ---------------------------------------------------------------------------

///|
test "gmres_solves_a_non_symmetric_system" {
  // Solve [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]] x = [8, -11, -3].
  // The exact solution is [2, 3, -1].
  let a = mkmat([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
  let x = match
    @moongrad.gmres(a, [8.0, -11.0, -3.0], [0.0, 0.0, 0.0], 20, 1.0e-12) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  check(near(x[0], 2.0, 1.0e-8), "x0 should be 2, got \{x[0]}")
  check(near(x[1], 3.0, 1.0e-8), "x1 should be 3, got \{x[1]}")
  check(near(x[2], -1.0, 1.0e-8), "x2 should be -1, got \{x[2]}")
}

///|
test "gmres_restarts_when_the_krylov_space_closes" {
  // A four by four system whose Krylov space is exhausted after four
  // vectors, so the iteration has to restart to finish the job. The
  // exact solution is [1, 2, 3, 4].
  let a = mkmat([
    [4.0, 1.0, 0.0, 0.0],
    [-1.0, 4.0, 1.0, 0.0],
    [0.0, -1.0, 4.0, 1.0],
    [0.0, 0.0, -1.0, 4.0],
  ])
  let b = [6.0, 10.0, 14.0, 13.0]
  let x = match @moongrad.gmres(a, b, [0.0, 0.0, 0.0, 0.0], 25, 1.0e-12) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  let want = [1.0, 2.0, 3.0, 4.0]
  for i = 0; i < 4; i = i + 1 {
    check(
      near(x[i], want[i], 1.0e-8),
      "x\{i} should be \{want[i]}, got \{x[i]}",
    )
  }
}

///|
test "gmres_refuses_a_dimension_mismatch" {
  let a = mkmat([[2.0, 1.0], [1.0, 3.0]])
  let x = @moongrad.gmres(a, [1.0, 2.0, 3.0], [0.0, 0.0], 10, 1.0e-10)
  let refused = match x {
    None => true
    Some(_) => false
  }
  check(refused, "a right-hand side of the wrong length has no solution")
}

///|
test "gmres_refuses_a_non_square_matrix" {
  let a = mkmat([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0]])
  let x = @moongrad.gmres(a, [1.0, 2.0], [0.0, 0.0], 10, 1.0e-10)
  let refused = match x {
    None => true
    Some(_) => false
  }
  check(refused, "a non-square matrix has no GMRES solve")
}

// ---------------------------------------------------------------------------
// The SPD convenience wrapper
// ---------------------------------------------------------------------------

///|
test "cg_solve_spd_agrees_with_conj_grad" {
  // The wrapper is conjugate gradients from zero with a budget of
  // four times the dimension, so it has to reproduce the same call.
  let a = mkmat([[4.0, 1.0], [1.0, 3.0]])
  let b = [1.0, 2.0]
  let wrapped = match @moongrad.cg_solve_spd(a, b) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  let direct = match @moongrad.conj_grad(a, b, [0.0, 0.0], 8, 1.0e-10) {
    Some(v) => v
    None => {
      check(false, "the system should be solvable")
      return
    }
  }
  check(wrapped.length() == direct.length(), "both should have two entries")
  for i = 0; i < wrapped.length(); i = i + 1 {
    check(
      near(wrapped[i], direct[i], 1.0e-12),
      "the two solves should agree at x\{i}",
    )
  }
  check(
    residual_of(a, wrapped, b) < 1.0e-9,
    "the residual should be small, got \{residual_of(a, wrapped, b)}",
  )
}

// ---------------------------------------------------------------------------
// The Jacobi eigendecomposition
// ---------------------------------------------------------------------------

///|
test "jacobi_finds_the_eigenvalues_of_a_two_by_two_matrix" {
  // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
  let a = mkmat([[2.0, 1.0], [1.0, 2.0]])
  let (eigenvalues, vectors) = match
    @moongrad.symmetric_eigen_jacobi_all(a, 50) {
    Some(pair) => pair
    None => {
      check(false, "the decomposition should exist")
      return
    }
  }
  check(eigenvalues.length() == 2, "there should be two eigenvalues")
  check(
    near(eigenvalues[0], 1.0, 1.0e-10),
    "the smaller eigenvalue should be 1, got \{eigenvalues[0]}",
  )
  check(
    near(eigenvalues[1], 3.0, 1.0e-10),
    "the larger eigenvalue should be 3, got \{eigenvalues[1]}",
  )
  // v diag(lambda) v^T has to be the original matrix.
  for i = 0; i < 2; i = i + 1 {
    for j = 0; j < 2; j = j + 1 {
      let mut acc = 0.0
      for k = 0; k < 2; k = k + 1 {
        acc = acc + vectors.get(i, k) * eigenvalues[k] * vectors.get(j, k)
      }
      check(
        near(acc, a.get(i, j), 1.0e-10),
        "the reconstruction should match at (\{i}, \{j}), got \{acc}",
      )
    }
  }
}

///|
test "jacobi_finds_the_eigenvalues_of_a_three_by_three_matrix" {
  let a = mkmat([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
  let (eigenvalues, vectors) = match
    @moongrad.symmetric_eigen_jacobi_all(a, 50) {
    Some(pair) => pair
    None => {
      check(false, "the decomposition should exist")
      return
    }
  }
  check(eigenvalues.length() == 3, "there should be three eigenvalues")
  // The eigenvalues of a symmetric matrix sum to its trace and
  // multiply to its determinant.
  let mut total = 0.0
  for e in eigenvalues {
    total = total + e
  }
  check(near(total, 9.0, 1.0e-9), "the trace is 9, got \{total}")
  let mut product = 1.0
  for e in eigenvalues {
    product = product * e
  }
  check(near(product, 18.0, 1.0e-9), "the determinant is 18, got \{product}")
  for i = 0; i < 3; i = i + 1 {
    for j = 0; j < 3; j = j + 1 {
      let mut acc = 0.0
      for k = 0; k < 3; k = k + 1 {
        acc = acc + vectors.get(i, k) * eigenvalues[k] * vectors.get(j, k)
      }
      check(
        near(acc, a.get(i, j), 1.0e-10),
        "the reconstruction should match at (\{i}, \{j}), got \{acc}",
      )
    }
  }
}

///|
test "jacobi_returns_the_eigenvalues_in_ascending_order" {
  let a = mkmat([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
  let (eigenvalues, _vectors) = match
    @moongrad.symmetric_eigen_jacobi_all(a, 50) {
    Some(pair) => pair
    None => {
      check(false, "the decomposition should exist")
      return
    }
  }
  check(
    near(eigenvalues[0], 1.0, 1.0e-10),
    "the smallest eigenvalue should be 1, got \{eigenvalues[0]}",
  )
  check(
    near(eigenvalues[1], 2.0, 1.0e-10),
    "the middle eigenvalue should be 2, got \{eigenvalues[1]}",
  )
  check(
    near(eigenvalues[2], 3.0, 1.0e-10),
    "the largest eigenvalue should be 3, got \{eigenvalues[2]}",
  )
}

///|
test "jacobi_reconstructs_a_five_by_five_matrix_from_a_small_budget" {
  // Six sweeps is the budget the other Jacobi routine in the library
  // uses, and it is enough here too.
  let a = mkmat([
    [4.0, -1.0, 0.0, 0.0, 0.0],
    [-1.0, 4.0, -1.0, 0.0, 0.0],
    [0.0, -1.0, 4.0, -1.0, 0.0],
    [0.0, 0.0, -1.0, 4.0, -1.0],
    [0.0, 0.0, 0.0, -1.0, 4.0],
  ])
  let (eigenvalues, vectors) = match
    @moongrad.symmetric_eigen_jacobi_all(a, 6) {
    Some(pair) => pair
    None => {
      check(false, "the decomposition should exist")
      return
    }
  }
  check(eigenvalues.length() == 5, "there should be five eigenvalues")
  for i = 0; i < 5; i = i + 1 {
    for j = 0; j < 5; j = j + 1 {
      let mut acc = 0.0
      for k = 0; k < 5; k = k + 1 {
        acc = acc + vectors.get(i, k) * eigenvalues[k] * vectors.get(j, k)
      }
      check(
        near(acc, a.get(i, j), 1.0e-10),
        "the reconstruction should match at (\{i}, \{j}), got \{acc}",
      )
    }
  }
}

///|
test "jacobi_refuses_a_non_square_matrix" {
  let a = mkmat([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0]])
  let decomposition = @moongrad.symmetric_eigen_jacobi_all(a, 50)
  let refused = match decomposition {
    None => true
    Some(_) => false
  }
  check(refused, "a non-square matrix has no eigen decomposition")
}

// ---------------------------------------------------------------------------
// The symmetric part
// ---------------------------------------------------------------------------

///|
test "symmetric_part_averages_a_matrix_with_its_transpose" {
  let a = mkmat([[1.0, 2.0], [4.0, 6.0]])
  let s = @moongrad.symmetric_part(a)
  check(near(s.get(0, 0), 1.0, 1.0e-12), "the diagonal is unchanged")
  check(near(s.get(1, 1), 6.0, 1.0e-12), "the diagonal is unchanged")
  check(
    near(s.get(0, 1), 3.0, 1.0e-12),
    "the off-diagonal should be the average, got \{s.get(0, 1)}",
  )
  check(
    near(s.get(1, 0), 3.0, 1.0e-12),
    "the off-diagonal should be the average, got \{s.get(1, 0)}",
  )
}

///|
test "symmetric_part_of_a_symmetric_matrix_is_the_matrix" {
  let a = mkmat([[2.0, 1.0], [1.0, 3.0]])
  let s = @moongrad.symmetric_part(a)
  for i = 0; i < 2; i = i + 1 {
    for j = 0; j < 2; j = j + 1 {
      check(
        near(s.get(i, j), a.get(i, j), 1.0e-12),
        "the entry at (\{i}, \{j}) should be unchanged",
      )
    }
  }
}

// ---------------------------------------------------------------------------
// The spectral norm
// ---------------------------------------------------------------------------

///|
test "spectral_norm_bound_of_the_identity_is_one" {
  let a = @moongrad.Matrix::identity(3)
  let bound = @moongrad.spectral_norm_bound(a)
  check(bound >= 1.0, "the bound should be at least 1, got \{bound}")
  check(near(bound, 1.0, 1.0e-6), "the bound should be 1, got \{bound}")
}

///|
test "spectral_norm_bound_of_a_scaled_identity_is_the_scale" {
  let a = mkmat([[3.0, 0.0], [0.0, 3.0]])
  let bound = @moongrad.spectral_norm_bound(a)
  check(near(bound, 3.0, 1.0e-6), "the bound should be 3, got \{bound}")
}

///|
test "spectral_norm_bound_of_a_rank_one_matrix_is_its_largest_singular_value" {
  // [[1, 2], [2, 4]] has the singular values 5 and 0.
  let a = mkmat([[1.0, 2.0], [2.0, 4.0]])
  let bound = @moongrad.spectral_norm_bound(a)
  check(near(bound, 5.0, 1.0e-6), "the bound should be 5, got \{bound}")
}
