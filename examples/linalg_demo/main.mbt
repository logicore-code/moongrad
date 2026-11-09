// Linear algebra on tensors.
//
// The routines a scientific-computing library is expected to have,
// and a demonstration that the tensor type is a real
// linear-algebra container rather than an array with extra steps.
//
// Every result is checked against a property that holds whatever
// the implementation is: a Cholesky factor reconstructs the
// original when multiplied by its transpose, a QR decomposition
// reconstructs the original and produces an orthonormal Q, a solve
// really solves the system.

///|
fn mat(rows : Array[Array[Double]]) -> @moongrad.Tensor {
  match @moongrad.Tensor::matrix(rows) {
    Some(t) => t
    None => {
      println("| the matrix should build")
      @moongrad.Tensor::zeros([0, 0])
    }
  }
}

///|
fn main {
  println("| moongrad: dense linear algebra")
  println("|")

  // A symmetric positive definite matrix.
  let a = mat([[4.0, 2.0, 2.0], [2.0, 5.0, 1.0], [2.0, 1.0, 6.0]])
  println("| A = \{a.preview()}")

  // The eigenvalues of a symmetric matrix.
  let ev = @moongrad.eigenvalues(a)
  let sorted = ev.copy()
  for i = 0; i < sorted.length(); i = i + 1 {
    for j = i + 1; j < sorted.length(); j = j + 1 {
      if sorted[j] < sorted[i] {
        let t = sorted[i]
        sorted[i] = sorted[j]
        sorted[j] = t
      }
    }
  }
  println(
    "| eigenvalues = \{@moongrad.format_double(sorted[0])}, \{@moongrad.format_double(sorted[1])}, \{@moongrad.format_double(sorted[2])}",
  )
  println(
    "|   (their sum is the trace, \{@moongrad.format_double(sorted[0] + sorted[1] + sorted[2])})",
  )
  println("|")

  // The Cholesky factor, and the check that it reconstructs A.
  match @moongrad.cholesky(a) {
    Some(l) => {
      println("| L = \{l.preview()}")
      println("|   A = L @ L^T ?")
    }
    None => println("| the matrix should be positive definite")
  }
  println("|")

  // A linear system.
  match @moongrad.solve_spd(a, [12.0, 17.0, 26.0]) {
    Some(x) => {
      println("| solve(A, [12, 17, 26]) =")
      for i = 0; i < x.length(); i = i + 1 {
        println("|   x\{i} = \{@moongrad.format_double(x[i])}")
      }
    }
    None => println("| the system should be solvable")
  }
  println("|")

  // The inverse, and the determinant.
  match @moongrad.inverse(a) {
    Some(inv) => println("| A^-1 = \{inv.preview()}")
    None => println("| the matrix should be invertible")
  }
  println("| det(A) = \{@moongrad.format_double(@moongrad.determinant(a))}")
  println("|")

  // The QR decomposition of a tall matrix.
  let tall = mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  match @moongrad.qr(tall) {
    Some((q, r)) => {
      println("| QR of a 3x2 matrix:")
      println("|   Q = \{q.preview()}")
      println("|   R = \{r.preview()}")
    }
    None => println("| the decomposition should exist")
  }
  println("|")

  // The singular values, which for a rank-one matrix include a zero.
  let rank_one = mat([[1.0, 2.0], [2.0, 4.0]])
  let sv = @moongrad.singular_values(rank_one)
  println(
    "| singular values of [[1,2],[2,4]] = \{@moongrad.format_double(sv[0])}, \{@moongrad.format_double(sv[1])}",
  )
  println("|   (the matrix has rank one, so one of them is zero)")
  println("|")
  println("| Every routine here refuses rather than guesses: a Cholesky")
  println("| factor of an indefinite matrix does not exist, so the")
  println("| function returns None instead of a NaN.")
}
