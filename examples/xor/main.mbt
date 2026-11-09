// XOR.
//
// The classic problem a linear model cannot solve and a two-layer
// network can. The four points form two clusters that no straight
// line separates, so a single layer's decision boundary — a line —
// gets at most two of the four right.
//
// The run prints the loss curve and the four predictions, so the
// transition from "the network cannot fit this" to "the network
// fits this" is visible rather than asserted.
//
// The interesting part is the *loss surface*. XOR is the standard
// example of a non-convex objective: with a squared error and no
// nonlinearity at all, the loss has a saddle at the origin, and
// gradient descent starting from a symmetric initialisation stays
// there forever. Adding the hidden layer with a nonlinearity is
// what makes the problem separable, and the example shows that
// directly by contrasting the two.

///|
let xs : Array[Array[Double]] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

///|
let ys : Array[Double] = [0.0, 1.0, 1.0, 0.0]

///|
fn main raise @moongrad.ADError {
  let data = match @moongrad.Dataset::new(xs, ys, 1) {
    Some(d) => d
    None => {
      println("| the dataset is malformed")
      return
    }
  }
  println("| moongrad: XOR with a two-layer network")
  println("|")
  // First: a linear model, which provably cannot do this.
  let linear = @moongrad.Trainer::new(
    [2, 1],
    [@moongrad.Activation::Identity],
    data,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    7,
  )
  for i = 0; i < 600; i = i + 1 {
    ignore(linear.train_step(-1.0))
  }
  println(
    "| linear model  (2 -> 1, identity): final loss = \{@moongrad.format_double(linear.final_loss())}",
  )
  println("|")
  // Then the same data with a hidden layer and a nonlinearity.
  //
  // The output layer is linear and the cross-entropy is computed
  // from the logit, which is the numerically stable pairing:
  // softplus(z) - z*y. Applying a sigmoid first and then feeding
  // the probability into the loss would squash the gradient exactly
  // where it needs to be largest.
  let net = @moongrad.Trainer::new(
    [2, 8, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    data,
    @moongrad.LossKind::Bce,
    @moongrad.Optimizer::Adam(0.05),
    11,
  )
  let first = net.train_step(-1.0)
  println("| step 0     loss = \{@moongrad.format_double(first.loss)}")
  for i = 1; i <= 1500; i = i + 1 {
    let r = net.train_step(-1.0)
    if i % 250 == 0 {
      println(
        "| step \{i}    loss = \{@moongrad.format_double(r.loss)}   |grad| = \{@moongrad.format_double(r.grad_norm)}",
      )
    }
  }
  println("| final loss = \{@moongrad.format_double(net.final_loss())}")
  println("|")
  // Evaluate the trained network on the four points.
  println("| input      expected   predicted")
  let g = net.graph
  let params = net.net.params()
  let flat : Array[Double] = []
  for p in params {
    let arr = p.to_array()
    for v in arr {
      flat.push(v)
    }
  }
  let b = @moongrad.trainer_binding(g, data, flat, net.net)
  for s = 0; s < 4; s = s + 1 {
    let input : Array[Int32] = []
    for f = 0; f < 2; f = f + 1 {
      input.push(g.variable(s * 2 + f))
    }
    let logit = net.net.apply(g, input)[0]
    let z = @moongrad.forward(g, b, logit).output
    let v = 1.0 / (1.0 + @math.exp(-z))
    println(
      "| (\{data.samples[s].x[0]}, \{data.samples[s].x[1]})   \{@moongrad.format_double(data.samples[s].y)}        \{@moongrad.format_double(v)}",
    )
  }
  println("|")
  println("| A linear model cannot separate XOR: its decision boundary")
  println("| is a line, and no line separates two diagonally opposite")
  println("| corners of a square from the other two. The hidden layer")
  println("| with a nonlinearity is what makes the problem separable.")
}
