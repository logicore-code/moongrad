// Second derivatives on the tensor side.
//
// A gradient that is a program can be differentiated again. That is
// the whole reason the backward sweep emits expressions instead of
// numbers: the second derivative does not need a second engine, it
// needs the first engine pointed at a different node.
//
// This example builds a tensor objective, asks for its Hessian, and
// checks the answer three ways: against the closed form, against a
// finite difference of the gradient, and by taking a Newton step and
// watching the objective fall to zero.

///|
/// f(x) = sum_i (x_i - c_i)^2, whose Hessian is 2 I.
fn build_quadratic(c : Array[Double]) -> @moongrad.TGraph {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let n = c.length()
  let mut total = g.constant(@moongrad.Tensor::filled([], 0.0))
  for i = 0; i < n; i = i + 1 {
    // One element at a time: the graph works on whole tensors, so
    // the coordinate is picked out with a slice.
    let xi = g.slice(x, 0, i, 1)
    let centre = g.constant(@moongrad.Tensor::filled([], c[i]))
    let d = g.sub(xi, centre)
    let sq = g.mul(d, d)
    total = g.add(total, g.sum_axis(sq, 0))
  }
  g
}

///|
fn binding_of(vals : Array[Double]) -> @moongrad.TBinding {
  let b = @moongrad.TBinding::new()
  let t = @moongrad.Tensor::zeros([vals.length()])
  for i = 0; i < vals.length(); i = i + 1 {
    t.data[i] = vals[i]
  }
  b.set_input(0, t)
  b
}

///|
fn value_of(g : @moongrad.TGraph, vals : Array[Double]) -> Double {
  let values = @moongrad.t_forward(g, binding_of(vals))
  values[values.length() - 1].data[0]
}

///|
fn bracket(items : Array[String]) -> String {
  "[" + items.join(", ") + "]"
}

///|
fn main {
  println("| moongrad: the Hessian of a tensor objective")
  println("|")
  println("| f(x) = sum_i (x_i - c_i)^2")
  println("|")
  println("| The Hessian of a quadratic is constant, so it is also the")
  println("| Jacobian of the gradient — which is the useful fact: a")
  println("| Newton step is one linear solve, and the engine gets it")
  println("| by differentiating the gradient it already compiled.")
  println("|")
  let c = [1.0, -2.0, 0.5]
  let g = build_quadratic(c)
  let x0 = [4.0, 1.0, -3.0]
  let b = binding_of(x0)
  let target = g.nodes.length() - 1
  let h = @moongrad.t_hessian(g, b, target)
  let x0_texts : Array[String] = []
  for i = 0; i < x0.length(); i = i + 1 {
    x0_texts.push(@moongrad.format_double(x0[i]))
  }
  let x0_text = bracket(x0_texts)
  println("| at x = " + x0_text)
  println("|")
  println("| Hessian")
  println("| ----------------------")
  for i = 0; i < h.nrows(); i = i + 1 {
    let row : Array[String] = []
    for j = 0; j < h.ncols(); j = j + 1 {
      row.push(@moongrad.format_double(h.get(i, j)))
    }
    println("|   " + bracket(row))
  }
  println("|")
  // The closed form: 2 I.
  let mut max_error = 0.0
  for i = 0; i < h.nrows(); i = i + 1 {
    for j = 0; j < h.ncols(); j = j + 1 {
      let want = if i == j { 2.0 } else { 0.0 }
      let diff = h.get(i, j) - want
      if diff > max_error {
        max_error = diff
      }
      if -diff > max_error {
        max_error = -diff
      }
    }
  }
  println(
    "| largest disagreement with the closed form 2I: \{@moongrad.format_double_exact(max_error)}",
  )
  // The finite difference of the gradient, which is an independent
  // derivation of the same object.
  let eps = 1.0e-5
  let mut fd_error = 0.0
  for j = 0; j < x0.length(); j = j + 1 {
    let up = x0.copy()
    let down = x0.copy()
    up[j] = up[j] + eps
    down[j] = down[j] - eps
    let gu = @moongrad.t_grad(g, binding_of(up), target)
    let gd = @moongrad.t_grad(g, binding_of(down), target)
    for i = 0; i < x0.length(); i = i + 1 {
      let numeric = (gu.input_grads[0].data[i] - gd.input_grads[0].data[i]) /
        (2.0 * eps)
      let diff = h.get(i, j) - numeric
      if diff > fd_error {
        fd_error = diff
      }
      if -diff > fd_error {
        fd_error = -diff
      }
    }
  }
  println(
    "| largest disagreement with a finite difference: \{@moongrad.format_double(fd_error)}",
  )
  // A Newton step: solve H dx = -g and move along dx. The Hessian
  // of this objective is positive definite, so the Cholesky solver
  // applies.
  let grad = @moongrad.t_grad(g, b, target)
  let rhs : Array[Double] = []
  for i = 0; i < x0.length(); i = i + 1 {
    rhs.push(-grad.input_grads[0].data[i])
  }
  let dx = match @moongrad.cholesky_solve(h, rhs) {
    Some(v) => v
    None => {
      println("| the Cholesky solve failed")
      return
    }
  }
  let x1 = x0.copy()
  for i = 0; i < x0.length(); i = i + 1 {
    x1[i] = x1[i] + dx[i]
  }
  let before = value_of(g, x0)
  let after = value_of(g, x1)
  println("|")
  println("| one Newton step")
  println("|   objective before: \{@moongrad.format_double(before)}")
  println("|   objective after:  \{@moongrad.format_double(after)}")
  let xs1 : Array[String] = []
  for i = 0; i < x1.length(); i = i + 1 {
    xs1.push(@moongrad.format_double(x1[i]))
  }
  println("|   x after the step: " + bracket(xs1))
  println("|")
  println("| The step lands on the minimiser, because the Hessian of a")
  println("| quadratic is exact rather than approximated.")
}
