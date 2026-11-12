// A gradient that outlives its input.
//
// A tape-based backward pass produces a number. A gradient program
// produces an expression, and an expression can be evaluated at
// whatever the input happens to be when it is needed.
//
// The difference only shows up when a derivative is not a smooth
// function of its input. ReLU's derivative is a step: it is one
// where the input is positive and zero where it is negative. A
// backward pass that folds that branch into a constant at compile
// time is right at the point it was compiled and wrong at every
// other point — and it fails silently, because the wrong number
// looks exactly like the right one.
//
// This example compiles a gradient once and then evaluates it at
// four different inputs, including two that sit on the other side of
// the kink. The gradient follows the input every time.

///|
/// f(x) = sum_i relu(x_i) * x_i.
///
/// The derivative is 2 x_i where x_i is positive and 0 where it is
/// negative, so the kink shows up in the answer.
fn build() -> @moongrad.TGraph {
  let g = @moongrad.TGraph::new()
  let x = g.variable(0)
  let r = g.relu(x)
  let y = g.mul(r, x)
  ignore(g.sum_axis(y, 0))
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
/// f(x) evaluated directly, for the finite-difference check.
fn value_of(vals : Array[Double]) -> Double {
  let mut total = 0.0
  for i = 0; i < vals.length(); i = i + 1 {
    if vals[i] > 0.0 {
      total = total + vals[i] * vals[i]
    }
  }
  total
}

///|
fn numeric_grad(vals : Array[Double]) -> Array[Double] {
  let h = 1.0e-6
  let out : Array[Double] = []
  for i = 0; i < vals.length(); i = i + 1 {
    let xp = vals.copy()
    let xm = vals.copy()
    xp[i] = xp[i] + h
    xm[i] = xm[i] - h
    out.push((value_of(xp) - value_of(xm)) / (2.0 * h))
  }
  out
}

///|
/// Pad a string on the right to a fixed width.
fn pad(s : String, width : Int32) -> String {
  let mut out = s
  while out.length() < width {
    out = out + " "
  }
  out
}

///|
fn main {
  println("| moongrad: one gradient, many inputs")
  println("|")
  println("| f(x) = sum_i relu(x_i) * x_i, so df/dx_i is 2 x_i")
  println("| where x_i is positive and 0 where it is negative.")
  println("|")
  println("| The gradient is compiled once, from the first input")
  println("| below, and then evaluated at every input.")
  println("|")
  let g = build()
  let compiled = @moongrad.compile_tensor_gradient(
    g,
    binding_of([2.0, -1.0, 0.5]),
    2,
  )
  let points : Array[Array[Double]] = [
    [2.0, -1.0, 0.5],
    [-2.0, -1.0, -0.5],
    [2.0, 3.0, 0.5],
    [-4.0, 1.5, -0.25],
  ]
  println("|   x                      compiled gradient      finite difference")
  println("| -------------------------------------------------------")
  let mut worst = 0.0
  for p in points {
    let (grads, _) = @moongrad.evaluate_tensor_gradient(compiled, binding_of(p))
    let numeric = numeric_grad(p)
    let parts : Array[String] = []
    let numbers : Array[String] = []
    let mut agrees = true
    for i = 0; i < p.length(); i = i + 1 {
      parts.push(@moongrad.format_double(grads[0].data[i]))
      numbers.push(@moongrad.format_double(numeric[i]))
      let diff = grads[0].data[i] - numeric[i]
      let magnitude = if diff < 0.0 { -diff } else { diff }
      if magnitude > worst {
        worst = magnitude
      }
      if diff > 1.0e-5 || diff < -1.0e-5 {
        agrees = false
      }
    }
    let xs : Array[String] = []
    for i = 0; i < p.length(); i = i + 1 {
      xs.push(@moongrad.format_double(p[i]))
    }
    let flag = if agrees { "" } else { "   MISMATCH" }
    let xs_text = "[" + xs.join(", ") + "]"
    let parts_text = "[" + parts.join(", ") + "]"
    let numbers_text = "[" + numbers.join(", ") + "]"
    let line = "| " +
      pad(xs_text, 24) +
      pad(parts_text, 24) +
      numbers_text +
      flag
    println(line)
  }
  println("|")
  println("| The largest disagreement with the finite difference is")
  println("| \{@moongrad.format_double(worst)}, which is the finite")
  println("| difference's own error rather than the gradient's.")
  println("|")
  println("| Note what happens on the second row: the gradient was")
  println("| compiled where x0 was positive, and it reports 0 there")
  println("| because that is the derivative at the new input. A")
  println("| gradient frozen at its compile point would report 4.")
}
