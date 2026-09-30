// hessian_test.mbt
//
// Second-order derivatives and the Newton solver built on them.
//
// The pattern is the same as the core tests: a closed form first,
// then a cross-check that the two derivations agree.

///|
fn hess_of(f_of : (Graph, Int, Int) -> Int raise ADError, xs : Array[Double]) -> Matrix raise ADError {
  let g = Graph::new()
  ignore(g.new_vars(xs.length()))
  let f = f_of(g, g.variable(0), g.variable(1))
  hessian(g, binding_of(xs), f)
}

///|
test "hessian_of_a_quadratic_is_constant" {
  // f(x, y) = x^2 + x*y + y^2.
  // H = [[2, 1], [1, 2]].
  let h = hess_of(
    fn(g : Graph, x : Int, y : Int) -> Int raise ADError {
      g.add(g.add(g.pow(x, 2.0), g.mul(x, y)), g.pow(y, 2.0))
    },
    [3.0, 4.0],
  )
  assert_close(h.get(0, 0), 2.0, 1.0e-9, "d2f/dx2")
  assert_close(h.get(0, 1), 1.0, 1.0e-9, "d2f/dxdy")
  assert_close(h.get(1, 0), 1.0, 1.0e-9, "d2f/dydx")
  assert_close(h.get(1, 1), 2.0, 1.0e-9, "d2f/dy2")
}

///|
test "hessian_of_a_cubic_matches_the_closed_form" {
  // f(x, y) = x^3 + y^2 * x.
  // d2f/dx2 = 6x, d2f/dxdy = 2y, d2f/dy2 = 2x.
  let h = hess_of(
    fn(g : Graph, x : Int, y : Int) -> Int raise ADError {
      g.add(g.pow(x, 3.0), g.mul(g.pow(y, 2.0), x))
    },
    [2.0, 5.0],
  )
  assert_close(h.get(0, 0), 12.0, 1.0e-9, "d2f/dx2 = 6x")
  assert_close(h.get(0, 1), 10.0, 1.0e-9, "d2f/dxdy = 2y")
  assert_close(h.get(1, 1), 4.0, 1.0e-9, "d2f/dy2 = 2x")
}

///|
test "hessian_of_a_transcendental_matches_the_closed_form" {
  // f(x, y) = exp(x*y).
  //   df/dx     = y e^(xy)          d2f/dx2   = y^2 e^(xy)
  //   d2f/dxdy  = (1 + xy) e^(xy)   d2f/dy2   = x^2 e^(xy)
  // The off-diagonal is the interesting one: the product rule
  // contributes a plain 1 alongside the exponential, and an engine
  // that only propagates through the exponential gets it wrong.
  let h = hess_of(
    fn(g : Graph, x : Int, y : Int) -> Int raise ADError {
      g.exp(g.mul(x, y))
    },
    [0.5, 1.5],
  )
  let e = @math.exp(0.75)
  assert_close(h.get(0, 0), 1.5 * 1.5 * e, 1.0e-9, "d2f/dx2")
  assert_close(h.get(0, 1), (1.0 + 0.5 * 1.5) * e, 1.0e-9, "d2f/dxdy")
  assert_close(h.get(1, 1), 0.5 * 0.5 * e, 1.0e-9, "d2f/dy2")
}

///|
test "hessian_agrees_with_a_numerical_second_difference" {
  // Central differences on the gradient are the third
  // implementation. Three independent derivations agreeing is
  // strong evidence; two agreeing is suggestive.
  let g = Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.exp(g.mul(x, x))
  let h = 1.0e-4
  let (gplus, _a) = grad(g, binding_of([1.0 + h]), f)
  let (gminus, _b) = grad(g, binding_of([1.0 - h]), f)
  let numerical = (gplus[0] - gminus[0]) / (2.0 * h)
  let analytic = hessian(g, binding_of([1.0]), f).get(0, 0)
  // f = e^(x^2), f'' = 2(1 + 2x^2) e^(x^2); at x = 1 that is 6e.
  assert_close(analytic, 6.0 * @math.exp(1.0), 1.0e-8, "analytic f''")
  assert_close(analytic, numerical, 1.0e-5, "analytic vs central difference")
}

///|
test "hessian_of_a_linear_function_is_zero" {
  let h = hess_of(
    fn(g : Graph, x : Int, y : Int) -> Int raise ADError {
      g.add(g.mul(x, g.constant(3.0)), g.mul(y, g.constant(-1.0)))
    },
    [5.0, 7.0],
  )
  assert_close(h.get(0, 0), 0.0, 1.0e-9, "d2f/dx2 of a linear function")
  assert_close(h.get(0, 1), 0.0, 1.0e-9, "d2f/dxdy of a linear function")
  assert_close(h.get(1, 1), 0.0, 1.0e-9, "d2f/dy2 of a linear function")
}

///|
test "hessian_of_a_disconnected_variable_row_is_zero" {
  let g = Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let unused = g.variable(1)
  let f = g.add(g.pow(x, 2.0), g.mul(unused, g.constant(0.0)))
  let h = hessian(g, binding_of([3.0, 9.0]), f)
  assert_close(h.get(0, 0), 2.0, 1.0e-9, "d2f/dx2")
  assert_close(h.get(1, 1), 0.0, 1.0e-9, "d2f/dy2 of an unused variable")
}

///|
test "hessian_symmetrisation_removes_asymmetric_noise" {
  let m = Matrix::zeros(2, 2)
  m.set(0, 0, 1.0)
  m.set(0, 1, 2.0)
  m.set(1, 0, 2.0000000001)
  m.set(1, 1, 3.0)
  let s = m.symmetrise()
  assert_close(s.get(0, 1), s.get(1, 0), 1.0e-15, "symmetry after averaging")
  assert_close(s.get(0, 1), 2.00000000005, 1.0e-15, "averaged off-diagonal")
}

// ---------------------------------------------------------------------------
// Jacobi eigenvalues
// ---------------------------------------------------------------------------

///|
test "eigenvalues_of_a_diagonal_matrix_are_its_diagonal" {
  let m = Matrix::zeros(3, 3)
  m.set(0, 0, 3.0)
  m.set(1, 1, 1.0)
  m.set(2, 2, 2.0)
  let ev = jacobi_eigenvalues(m)
  let sorted = sorted_copy(ev)
  assert_close(sorted[0], 1.0, 1.0e-9, "lambda_1")
  assert_close(sorted[1], 2.0, 1.0e-9, "lambda_2")
  assert_close(sorted[2], 3.0, 1.0e-9, "lambda_3")
}

///|
fn sorted_copy(xs : Array[Double]) -> Array[Double] {
  let out : Array[Double] = []
  for x in xs {
    out.push(x)
  }
  for i = 0; i < out.length(); i = i + 1 {
    for j = i + 1; j < out.length(); j = j + 1 {
      if out[j] < out[i] {
        let t = out[i]
        out[i] = out[j]
        out[j] = t
      }
    }
  }
  out
}

///|
test "eigenvalues_of_a_rotation_like_matrix" {
  // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
  let m = Matrix::zeros(2, 2)
  m.set(0, 0, 2.0)
  m.set(0, 1, 1.0)
  m.set(1, 0, 1.0)
  m.set(1, 1, 2.0)
  let ev = sorted_copy(jacobi_eigenvalues(m))
  assert_close(ev[0], 1.0, 1.0e-9, "lambda_min")
  assert_close(ev[1], 3.0, 1.0e-9, "lambda_max")
}

///|
test "hessian_of_a_strictly_convex_function_is_positive_definite" {
  // f(x, y) = x^2 + 4y^2 has eigenvalues 2 and 8, both positive,
  // which is the condition a Newton solver needs to take a full
  // step without damping.
  let h = hess_of(
    fn(g : Graph, x : Int, y : Int) -> Int raise ADError {
      g.add(g.pow(x, 2.0), g.mul(g.constant(4.0), g.pow(y, 2.0)))
    },
    [1.0, 1.0],
  )
  let ev = jacobi_eigenvalues(h)
  let smallest = minf(ev[0], ev[1])
  check(
    smallest > 0.0,
    "a strictly convex function has a positive Hessian, got \{ev}",
  )
}

// ---------------------------------------------------------------------------
// Cholesky
// ---------------------------------------------------------------------------

///|
test "cholesky_solves_a_known_system" {
  // A = [[4, 1], [1, 3]], b = [1, 2].
  // The exact solution is [1/11, 7/11].
  let a = Matrix::zeros(2, 2)
  a.set(0, 0, 4.0)
  a.set(0, 1, 1.0)
  a.set(1, 0, 1.0)
  a.set(1, 1, 3.0)
  let x = cholesky_solve(a, [1.0, 2.0])
  match x {
    Some(v) => {
      assert_close(v[0], 1.0 / 11.0, 1.0e-9, "x0")
      assert_close(v[1], 7.0 / 11.0, 1.0e-9, "x1")
    }
    None => check(false, "cholesky rejected a positive definite matrix")
  }
}

///|
test "cholesky_rejects_an_indefinite_matrix" {
  // [[1, 2], [2, 1]] is not positive definite.
  let a = Matrix::zeros(2, 2)
  a.set(0, 0, 1.0)
  a.set(0, 1, 2.0)
  a.set(1, 0, 2.0)
  a.set(1, 1, 1.0)
  let x = cholesky_solve(a, [1.0, 1.0])
  let rejected = match x {
    None => true
    Some(_) => false
  }
  check(rejected, "cholesky should reject an indefinite matrix")
}

// ---------------------------------------------------------------------------
// Newton
// ---------------------------------------------------------------------------

///|
test "newton_finds_the_exact_minimum_of_a_quadratic_in_one_step" {
  // Minimise f(x) = (x - 3)^2. The gradient is 2(x-3) and the
  // Hessian is 2, so one Newton step lands on 3 exactly.
  let g = Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let d = g.sub(x, g.constant(3.0))
  let f = g.mul(d, d)
  let (grads, _p) = grad(g, binding_of([0.0]), f)
  let h = hessian(g, binding_of([0.0]), f)
  let next = newton_step([0.0], grads, h, 0.0)
  match next {
    Some(v) => assert_close(v[0], 3.0, 1.0e-9, "one Newton step from 0")
    None => check(false, "newton_step failed on a convex quadratic")
  }
}

///|
test "newton_beats_gradient_descent_on_an_ill_conditioned_quadratic" {
  // f(x, y) = x^2 + 100 y^2. The condition number of the Hessian
  // is 100, and gradient descent needs on the order of that many
  // iterations to make progress along the narrow direction while
  // Newton needs one. This is the whole argument for second-order
  // methods in one test.
  let g = Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let f = g.add(g.pow(x, 2.0), g.mul(g.constant(100.0), g.pow(y, 2.0)))
  let start = [10.0, 0.1]
  let (grads, _p) = grad(g, binding_of(start), f)
  let h = hessian(g, binding_of(start), f)
  let newton = newton_step(start, grads, h, 0.0)
  match newton {
    Some(v) => {
      assert_close(v[0], 0.0, 1.0e-6, "Newton kills the wide direction")
      assert_close(v[1], 0.0, 1.0e-6, "Newton kills the narrow direction")
    }
    None => check(false, "newton_step failed on a positive definite quadratic")
  }
}

///|
test "newton_with_damping_still_descends" {
  let g = Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let d = g.sub(x, g.constant(3.0))
  let f = g.mul(d, d)
  let start = [0.0]
  let (grads, _p) = grad(g, binding_of(start), f)
  let h = hessian(g, binding_of(start), f)
  let next = newton_step(start, grads, h, 1.0)
  match next {
    Some(v) => {
      let f0 = forward(g, binding_of(start), f).output
      let f1 = forward(g, binding_of([v[0]]), f).output
      check(f1 < f0, "a damped Newton step must decrease the objective")
    }
    None => check(false, "damped newton_step should succeed on a convex quadratic")
  }
}

///|
test "newton_falls_back_when_the_hessian_is_singular" {
  // f(x, y) = x^2. The Hessian is diag(2, 0), not positive
  // definite, so a zero-damping Newton step must decline rather
  // than divide by zero. Damping makes it well posed.
  let g = Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let f = g.add(g.pow(x, 2.0), g.mul(y, g.constant(0.0)))
  let start = [5.0, 1.0]
  let (grads, _p) = grad(g, binding_of(start), f)
  let h = hessian(g, binding_of(start), f)
  let undamped = newton_step(start, grads, h, 0.0)
  let damped = newton_step(start, grads, h, 1.0)
  let refused = match undamped {
    None => true
    Some(_) => false
  }
  let accepted = match damped {
    None => false
    Some(_) => true
  }
  check(
    refused,
    "an undamped Newton step on a singular Hessian must be refused",
  )
  check(accepted, "damping should make the same step well posed")
}

///|
test "newton_converges_on_a_nonlinear_objective" {
  // f(x) = x^4 - 4x^3. f' = 4x^2(x-3), so x = 3 is the global
  // minimum and x = 0 is a local maximum. Newton from x = 5 has to
  // walk down a slope that is still rising at the start, which is
  // the case a naive fixed-step optimiser gets wrong.
  let g = Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.sub(g.pow(x, 4.0), g.mul(g.constant(4.0), g.pow(x, 3.0)))
  let mut current = 5.0
  let mut f_prev = @math.pow(5.0, 4.0) - 4.0 * @math.pow(5.0, 3.0)
  for _i = 0; _i < 12; _i = _i + 1 {
    let (grads, _p) = grad(g, binding_of([current]), f)
    let h = hessian(g, binding_of([current]), f)
    let next = newton_step([current], grads, h, 1.0e-9)
    match next {
      Some(v) => {
        let f_now = @math.pow(v[0], 4.0) - 4.0 * @math.pow(v[0], 3.0)
        // A tolerance rather than a strict inequality: at the
        // minimum f and f' both go to zero, so the last few steps
        // are inside the rounding noise.
        check(
          f_now < f_prev + 1.0e-9,
          "damped Newton must descend: \{f_prev} -> \{f_now}",
        )
        f_prev = f_now
        current = v[0]
      }
      None => {
        check(false, "damped Newton gave up on a well-behaved objective")
        break
      }
    }
  }
  check(
    absf(current - 3.0) < 1.0e-3,
    "Newton should reach the minimum at x = 3, got \{current}",
  )
}
