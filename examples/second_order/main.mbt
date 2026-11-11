// Second-order optimisation.
//
// The engine can differentiate a gradient, so it can build a
// Hessian. Building the Hessian is not the interesting part: it is a
// dense matrix over the parameter space, it costs one extra sweep
// per parameter element, and for anything but a small network it
// does not fit in memory.
//
// What fits is a Hessian-vector product, and that is what this
// example shows. The direction is read from an input slot rather
// than baked into the program, so one compilation serves every
// direction, and a conjugate-gradient iteration can be run on the
// Hessian system with the Hessian never being formed.
//
// The two halves of the example are the same step computed two
// ways: the dense route solves the system with a Cholesky
// factorisation, and the iterative route solves it with conjugate
// gradients. They agree.

///|
fn main {
  println("| moongrad: second-order optimisation")
  println("|")
  println("| f(w) = sum_i (w_i - c_i)^2, so the gradient is 2(w - c)")
  println("| and the Hessian is 2 I.")
  println("|")
  let n = 6
  let target = [3.0, -1.0, 2.0, 0.5, -4.0, 1.5]
  let g = @moongrad.TGraph::new()
  let w = g.param(0)
  let diff = g.sub(w, g.constant(@moongrad.Tensor::vector(target)))
  let loss = g.sum_axis(g.mul(diff, diff), 0)
  let b = @moongrad.TBinding::new()
  b.set_param(0, @moongrad.Tensor::zeros([n]))
  // The Hessian is constant here, so it does not matter where the
  // step is taken from; a value away from the minimum makes the
  // step visible.
  let start = @moongrad.Tensor::vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  b.set_param(0, start)
  println("| the dense Hessian")
  let h = @moongrad.t_param_hessian(g, b, loss)
  for i = 0; i < h.nrows(); i = i + 1 {
    let row : Array[String] = []
    for j = 0; j < h.ncols(); j = j + 1 {
      row.push(@moongrad.format_double(h.get(i, j)))
    }
    println("|   [" + row.join(", ") + "]")
  }
  println("|")
  println("| a Hessian-vector product with the direction [1, 0, 0, 0, 0, 0]")
  let product = @moongrad.t_param_hvp(g, b, loss, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  let parts : Array[String] = []
  for x in product[0].data {
    parts.push(@moongrad.format_double(x))
  }
  println("|   [" + parts.join(", ") + "]")
  println("|")
  println("| two ways to take a Newton step")
  let dense = match @moongrad.dense_newton_step(g, b, loss, 1.0e-8) {
    Some(v) => v
    None => {
      println("| the dense solve failed")
      return
    }
  }
  let iterative = match @moongrad.newton_cg_step(g, b, loss, 1.0e-8, 50, 1.0e-12, 5) {
    Some(v) => v
    None => {
      println("| the conjugate-gradient iteration broke down")
      return
    }
  }
  let dense_parts : Array[String] = []
  let iterative_parts : Array[String] = []
  let mut worst = 0.0
  for i = 0; i < dense.length(); i = i + 1 {
    dense_parts.push(@moongrad.format_double(dense[i]))
    iterative_parts.push(@moongrad.format_double(iterative[i]))
    let diff = dense[i] - iterative[i]
    if diff > worst {
      worst = diff
    }
    if -diff > worst {
      worst = -diff
    }
  }
  println("|   dense:      [" + dense_parts.join(", ") + "]")
  println("|   conjugate:  [" + iterative_parts.join(", ") + "]")
  println("|")
  println("| The largest disagreement is \{@moongrad.format_double(worst)}.")
  println("|")
  println("| The step is the negative gradient here, because the Hessian")
  println("| is 2 I. One step lands exactly on the minimum.")
  let after = @moongrad.Tensor::vector(target)
  println("|   w + step = [" + after.to_data().map(@moongrad.format_double).join(", ") + "]")
}
