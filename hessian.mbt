// hessian.mbt
//
// Second-order derivatives, eleven lines long.
//
// That is not a trick. The gradient is a graph, and `grad` is the
// only thing needed to differentiate a graph. So the Hessian entry
// H[i][j] is d/dx_j of the node the first sweep emitted for
// d f/d x_i — which is another call to the same compiler on the
// same graph, with a different target.
//
// Getting here is the reason the reverse pass emits expressions
// rather than numbers. A tape would have handed back 3.0 for
// d/dx x^3 and there would have been nothing left to
// differentiate; here the answer is a node, and a node is all
// higher-order machinery needs.
//
// The practical payoff is Newton's method. Gradient descent makes
// progress in proportion to the gradient's magnitude, so it crawls
// through long shallow valleys and stalls near a saddle; a Newton
// step uses the curvature information to rescale the step. For
// least-squares fitting in particular, Gauss-Newton and full Newton
// are the difference between "converges" and "converges in six
// iterations".

///|
/// Symmetrise: `(A + A^T) / 2`. A numerical Hessian built by two
/// separate differentiation passes can pick up asymmetry of order
/// the floating-point noise, and the noise is not harmless — it
/// moves the eigenvalues that a Newton step depends on.
pub fn Matrix::symmetrise(self : Matrix) -> Matrix {
  let n = self.rows
  let out = Matrix::zeros(n, n)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      out.set(i, j, 0.5 * (self.get(i, j) + self.get(j, i)))
    }
  }
  out
}

///|
/// The Hessian of the node `target` with respect to all variables,
/// evaluated at `b`.
///
/// Each row is produced by a separate `grad` call against the
/// gradient graph: the first sweep compiled d f / d x_i into a node,
/// and this sweeps the same graph again to compile
/// d^2 f / d x_j d x_i out of it. Nothing about the second sweep is
/// special-cased.
pub fn hessian(g : Graph, b : Binding, target : Int) -> Matrix raise ADError {
  let primal_values = forward_values(g, b)
  let first = compile_gradient(g, primal_values, target)
  let n = g.var_count
  let h = Matrix::zeros(n, n)
  for i = 0; i < n; i = i + 1 {
    // Differentiate the i-th gradient node with respect to every
    // variable, using the gradient graph as the primal this time.
    let (row, _p) = grad(first.graph, b, first.var_adj[i])
    for j = 0; j < n; j = j + 1 {
      h.set(i, j, row[j])
    }
  }
  h
}

///|
/// Total number of scalars across every parameter slot.
pub fn total_params(g : Graph) -> Int {
  let mut n = 0
  for i = 0; i < g.param_shapes.length(); i = i + 1 {
    n = n + g.param_shapes[i]
  }
  n
}

///|
/// Gauss-Newton step: solve `H delta = -g` for `delta` and return
/// `x + delta`.
///
/// `H` is expected to be symmetric positive definite, or close
/// enough; `damping` adds `damping * I` before factorising, which
/// is the standard Levenberg-Marquardt style fix for the case where
/// it is not. Setting `damping` to zero gives pure Gauss-Newton,
/// which is exact for a linear least-squares problem.
///
/// Returns `None` when the damped matrix is not positive definite,
/// so the caller can fall back to a gradient step rather than
/// producing a NaN.
pub fn newton_step(
  x : Array[Double],
  g : Array[Double],
  h : Matrix,
  damping : Double,
) -> Array[Double]? {
  let n = x.length()
  let a = h
  if damping != 0.0 {
    for i = 0; i < n; i = i + 1 {
      let d = a.get(i, i) + damping
      a.set(i, i, d)
    }
  }
  let solved = cholesky_solve(a, g)
  match solved {
    Some(delta) => {
      let out : Array[Double] = []
      for i = 0; i < n; i = i + 1 {
        out.push(x[i] - delta[i])
      }
      Some(out)
    }
    None => None
  }
}

///|
/// Solve `A y = b` for symmetric positive definite `A` by Cholesky
/// factorisation, in place over a scratch copy.
///
/// Returns `None` when a pivot goes non-positive, which is how a
/// non-convex region announces itself to a Newton solver.
pub fn cholesky_solve(a_in : Matrix, b : Array[Double]) -> Array[Double]? {
  let n = a_in.rows
  if n == 0 {
    return Some([])
  }
  // Lower-triangular L with A = L * L^T.
  let l = Matrix::zeros(n, n)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j <= i; j = j + 1 {
      let mut sum = a_in.get(i, j)
      for k = 0; k < j; k = k + 1 {
        sum = sum - l.get(i, k) * l.get(j, k)
      }
      if i == j {
        if sum <= 0.0 {
          return None
        }
        l.set(i, j, sum.sqrt())
      } else {
        l.set(i, j, sum / l.get(j, j))
      }
    }
  }
  // Forward substitution: L z = b
  let z : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    let mut sum = b[i]
    for k = 0; k < i; k = k + 1 {
      sum = sum - l.get(i, k) * z[k]
    }
    z.push(sum / l.get(i, i))
  }
  // Back substitution: L^T y = z. Solved rows are written into
  // their final index rather than appended, because the elimination
  // runs right to left and an append would store the answer in
  // reverse.
  let y : Array[Double] = []
  for _i = 0; _i < n; _i = _i + 1 {
    y.push(0.0)
  }
  for i = n - 1; i >= 0; i = i - 1 {
    let mut sum = z[i]
    for k = i + 1; k < n; k = k + 1 {
      sum = sum - l.get(k, i) * y[k]
    }
    y[i] = sum / l.get(i, i)
  }
  Some(y)
}

///|
/// Eigenvalues of a symmetric matrix, by the cyclic Jacobi method.
///
/// Newton solvers are usually driven by eigen-decomposition rather
/// than a Cholesky factorisation, and the test suite wants to check
// that the Hessian really does have the curvature the closed-form

///|
/// second derivative predicts. Jacobi is not the fastest option but
// it is short, unconditionally accurate for symmetric input, and
// has no failure mode to guard against.
pub fn jacobi_eigenvalues(a_in : Matrix) -> Array[Double] {
  let n = a_in.rows
  let a = Matrix::zeros(n, n)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      a.set(i, j, a_in.get(i, j))
    }
  }
  // Six sweeps is ample for the small matrices here; Jacobi
  // converges quadratically once the off-diagonal norm is small.
  for _sweep = 0; _sweep < 6; _sweep = _sweep + 1 {
    for p = 0; p < n - 1; p = p + 1 {
      for q = p + 1; q < n; q = q + 1 {
        let apq = a.get(p, q)
        if absf(apq) < 1.0e-14 {
          continue
        }
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
      }
    }
  }
  let out : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    out.push(a.get(i, i))
  }
  out
}
