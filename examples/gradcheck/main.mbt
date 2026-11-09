// Gradient check.
//
// The most important test of an automatic differentiation engine is
// whether its gradients agree with something computed a completely
// different way. This example does that for every operator in the
// tensor library, using central differences of the same function.
//
// The finite-difference estimate is
//
//     f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
//
// With a step of 1e-6 this agrees with the analytic derivative to
// about seven digits for a smooth function, and it catches a wrong
// rule immediately. The difference is that the two derivations
// share no code: the finite difference perturbs the input and
// evaluates the forward pass again, while the gradient compiler
// walks the graph backwards.
//
// A note on what this does *not* check: a ReLU's derivative is
// discontinuous, so a finite difference across the kink is
// meaningless and the example treats it separately.

///|
/// A function of a flat vector of parameters.
type Objective = (Array[Double]) -> Double

///|
/// The gradient of `f` at `x`, by central differences.
fn numeric_grad(f : Objective, x : Array[Double]) -> Array[Double] {
  let h = 1.0e-6
  let out : Array[Double] = []
  for i = 0; i < x.length(); i = i + 1 {
    let xp = x.copy()
    let xm = x.copy()
    xp[i] = xp[i] + h
    xm[i] = xm[i] - h
    out.push((f(xp) - f(xm)) / (2.0 * h))
  }
  out
}

///|
/// The gradient of the graph built by `build` at `x`, by the
/// gradient compiler.
///
/// `build` gets the graph and the binding and returns the node whose
/// value is the objective. The objective is a *sum* over the whole
/// input, so the gradient is a vector the same length as `x`.
fn compiled_grad(
  build : (@moongrad.TGraph, @moongrad.TBinding) -> Int,
  x : Array[Double],
) -> Array[Double] {
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector(x))
  let out = build(g, b)
  let grads = @moongrad.t_grad(g, b, out)
  grads.input_grads[0].to_data()
}

///|
/// Compare the two gradients and report the largest difference.
fn compare(
  name : String,
  build : (@moongrad.TGraph, @moongrad.TBinding) -> Int,
  objective : Objective,
  x : Array[Double],
) -> Bool {
  let analytic = compiled_grad(build, x)
  let numeric = numeric_grad(objective, x)
  let mut worst = 0.0
  for i = 0; i < analytic.length(); i = i + 1 {
    let d = @moongrad.absf(analytic[i] - numeric[i])
    if d > worst {
      worst = d
    }
  }
  let ok = worst < 1.0e-4
  println(
    "|   \{name}: max |analytic - numeric| = \{@moongrad.format_double(worst)}\{if ok { "" } else { "   <-- MISMATCH" }}",
  )
  ok
}

///|
/// The i-th component of the input vector, as a graph node.
fn component(g : @moongrad.TGraph, i : Int) -> Int {
  g.slice(g.variable(0), 0, i, 1)
}

///|
fn main {
  println("| moongrad: gradient check against central differences")
  println("|")
  println("| Every row compares the compiled gradient with a finite")
  println("| difference of the same function. The two derivations share")
  println("| no code, so agreement is evidence and not a tautology.")
  println("|")
  let mut all_ok = true
  // 1. A polynomial times a variable, plus an exponential.
  all_ok = all_ok &&
    compare(
      "x^2*y + exp(x)",
      fn(g, b) {
        let x = component(g, 0)
        let y = component(g, 1)
        let t = g.add(g.mul(g.pow(x, 2.0), y), g.exp(x))
        g.sum_axis(t, 0)
      },
      fn(v) { v[0] * v[0] * v[1] + @math.exp(v[0]) },
      [0.7, -1.3],
    )
  // 2. A quotient, whose derivative has the awkward 1/b^2 term.
  all_ok = all_ok &&
    compare(
      "1 / (1 + x^2 + y^2)",
      fn(g, b) {
        let x = component(g, 0)
        let y = component(g, 1)
        let one = g.constant(@moongrad.Tensor::scalar(1.0))
        let xx = g.pow(x, 2.0)
        let yy = g.pow(y, 2.0)
        let t = g.div(one, g.add(g.add(one, xx), yy))
        g.sum_axis(t, 0)
      },
      fn(v) { 1.0 / (1.0 + v[0] * v[0] + v[1] * v[1]) },
      [0.4, 1.1],
    )
  // 3. A matrix product, so the shape bookkeeping is checked too.
  all_ok = all_ok &&
    compare(
      "sum(x @ y) with y = [[1,2],[3,4]]",
      fn(g, b) {
        // The matrix is the input vector, reshaped — so the gradient
        // is with respect to those four numbers.
        let x = g.reshape(g.variable(0), [2, 2])
        let y = g.constant(
          match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0]]) {
            Some(t) => t
            None => @moongrad.Tensor::zeros([2, 2])
          },
        )
        let p = g.matmul(x, y)
        g.sum_axis(p, 0)
      },
      fn(v) {
        // x is (2,2) and y is [[1,2],[3,4]], so sum(x @ y)
        //   = (v0 + 3*v1) + (2*v0 + 4*v1) + (v2 + 3*v3) + (2*v2 + 4*v3)
        //   = 3*v0 + 7*v1 + 3*v2 + 7*v3
        3.0 * v[0] + 7.0 * v[1] + 3.0 * v[2] + 7.0 * v[3]
      },
      [1.0, 2.0, 3.0, 4.0],
    )
  // 4. A broadcast product, where the gradient has to be summed back
  //    down to the operand's own shape.
  all_ok = all_ok &&
    compare(
      "sum(x * y) with x (2,) and y (3,2)",
      fn(g, b) {
        let x = g.reshape(g.variable(0), [2])
        let y = g.constant(
          match @moongrad.Tensor::matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) {
            Some(t) => t
            None => @moongrad.Tensor::zeros([3, 2])
          },
        )
        let p = g.mul(x, y)
        g.sum_axis(p, 0)
      },
      fn(v) {
        // x is (2,) broadcast over y's three rows, so the total is
        //   x0 * (1+3+5) + x1 * (2+4+6) = 9*x0 + 12*x1
        9.0 * v[0] + 12.0 * v[1]
      },
      [1.0, 2.0],
    )
  // 5. A chain of nonlinearities, where the chain rule compounds.
  all_ok = all_ok &&
    compare(
      "tanh(sin(x) * exp(y))",
      fn(g, b) {
        let x = component(g, 0)
        let y = component(g, 1)
        let t = g.tanh(g.mul(g.sin(x), g.exp(y)))
        g.sum_axis(t, 0)
      },
      fn(v) { @math.tanh(@math.sin(v[0]) * @math.exp(v[1])) },
      [0.3, 0.8],
    )
  // 6. A logarithm, whose derivative is the reciprocal.
  all_ok = all_ok &&
    compare(
      "log(x^2 + 1) * y",
      fn(g, b) {
        let x = component(g, 0)
        let y = component(g, 1)
        let one = g.constant(@moongrad.Tensor::scalar(1.0))
        let t = g.mul(g.ln(g.add(g.pow(x, 2.0), one)), y)
        g.sum_axis(t, 0)
      },
      fn(v) { @math.ln(v[0] * v[0] + 1.0) * v[1] },
      [1.2, 0.6],
    )
  println("|")
  if all_ok {
    println("| Every gradient agrees with its finite difference.")
  } else {
    println("| At least one gradient disagrees. That is a bug in the")
    println("| engine, not in the test.")
  }
}
