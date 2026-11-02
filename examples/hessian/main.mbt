// Second-order derivatives.
//
// Builds a small function, differentiates it twice, and prints the
// Hessian next to the closed form. Then it uses the Hessian to take
// a Newton step and shows how far that gets in one iteration
// compared with how far gradient descent gets in the same number.
//
// The reason to care: gradient descent moves a distance set by the
// gradient's *magnitude*, so on a surface with curvature ratio 100
// it overshoots the narrow direction by two orders of magnitude and
// crawls through the wide one. A Newton step uses the curvature to
// rescale, and lands on the minimum in one step for a quadratic.
//
// The Hessian comes free from the design. The gradient is a graph,
// so the Hessian is just the same compiler run again with the
// gradient node as the target. There is no second-order code path
// to get wrong.

///|
fn main raise @moongrad.ADError {
  println("| moongrad: second-order derivatives")
  println("|")
  // f(x, y) = x^2 + x*y + y^2, whose Hessian is [[2, 1], [1, 2]].
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let f = g.add(g.add(g.pow(x, 2.0), g.mul(x, y)), g.pow(y, 2.0))
  let b = @moongrad.Binding::new()
  b.set_var(0, 1.3)
  b.set_var(1, -0.7)
  let h = @moongrad.hessian(g, b, f)
  println("| f(x, y) = x^2 + x*y + y^2")
  println("| Hessian at (1.3, -0.7):")
  for i = 0; i < 2; i = i + 1 {
    println(
      "|   [ \{@moongrad.format_double(h.get(i, 0))}  \{@moongrad.format_double(h.get(i, 1))} ]",
    )
  }
  println("|   exact: [ 2  1 ]")
  println("|           [ 1  2 ]")
  println("|")
  // Eigenvalues of the Hessian: 3 and 1. Both positive, so the
  // point is in a convex region and a Newton step is well posed.
  let ev = @moongrad.jacobi_eigenvalues(h)
  println(
    "| Hessian eigenvalues: \{@moongrad.format_double(ev[0])}, \{@moongrad.format_double(ev[1])}",
  )
  println("|")
  // A badly conditioned quadratic: f = x^2 + 100 y^2.
  let g2 = @moongrad.Graph::new()
  ignore(g2.new_vars(2))
  let u = g2.variable(0)
  let v = g2.variable(1)
  let q = g2.add(g2.pow(u, 2.0), g2.mul(g2.constant(100.0), g2.pow(v, 2.0)))
  let start = [10.0, 0.1]
  let b2 = @moongrad.Binding::new()
  b2.set_var(0, start[0])
  b2.set_var(1, start[1])
  let (grads, _p) = @moongrad.grad(g2, b2, q)
  let h2 = @moongrad.hessian(g2, b2, q)
  println("| f(x, y) = x^2 + 100 y^2, starting at (10, 0.1)")
  println(
    "|   gradient  = [\{@moongrad.format_double(grads[0])}, \{@moongrad.format_double(grads[1])}]",
  )
  println("|   Hessian   = [\{@moongrad.format_double(h2.get(0, 0))}  0 ]")
  println("|               [ 0  \{@moongrad.format_double(h2.get(1, 1))} ]")
  println(
    "|   condition number = \{@moongrad.format_double(h2.get(1, 1) / h2.get(0, 0))}",
  )
  println("|")
  // One Newton step against one gradient-descent step.
  let newton = @moongrad.newton_step(start, grads, h2, 0.0)
  let lr = 0.001
  let gd = [start[0] - lr * grads[0], start[1] - lr * grads[1]]
  let q_at = fn(p : Array[Double]) -> Double {
    p[0] * p[0] + 100.0 * p[1] * p[1]
  }
  println(
    "| objective at the start:      \{@moongrad.format_double(q_at(start))}",
  )
  match newton {
    Some(n) =>
      println(
        "| after one Newton step:      \{@moongrad.format_double(q_at(n))}   (at [\{@moongrad.format_double(n[0])}, \{@moongrad.format_double(n[1])}])",
      )
    None => println("| the Newton step was refused")
  }
  println(
    "| after one gradient step:    \{@moongrad.format_double(q_at(gd))}   (at [\{@moongrad.format_double(gd[0])}, \{@moongrad.format_double(gd[1])}])",
  )
  println("|")
  println("| Gradient descent takes a step whose length is set by the")
  println("| gradient's magnitude, so in the narrow direction it barely")
  println("| moves. The Newton step uses the curvature to rescale, and")
  println("| for a quadratic it lands on the minimum in one step.")
}
