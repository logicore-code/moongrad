// Neural network classifier.
//
// Trains a small multilayer perceptron on two well-separated
// clusters and reports the loss curve and the final predictions.
//
// Everything here goes through the tensor gradient compiler: the
// graph is built once, the gradient is compiled once, and every
// step re-evaluates the compiled gradient at the new weights. That
// is the whole point of emitting a gradient *program* rather than
// accumulating adjoints into a tape.
//
// The dataset is two clusters, which is a problem a linear model
// cannot solve — the decision boundary between them is not a line.
// With a hidden layer and a tanh nonlinearity the same model
// separates them, which is the difference the example is making.

///|
fn main raise @moongrad.ADError {
  println("| moongrad: a tensor neural network, trained by gradient descent")
  println("|")

  // Two clusters in the unit square, well separated along the
  // diagonal.
  let xs : Array[Array[Double]] = []
  let ys : Array[Double] = []
  for i = 0; i < 24; i = i + 1 {
    let t = i.to_double() / 23.0
    // The first cluster runs along (0.1, 0.1) to (0.3, 0.3).
    let a = 0.1 + 0.2 * t
    xs.push([a, a])
    ys.push(0.0)
    // The second runs along (0.7, 0.7) to (0.9, 0.9).
    let b = 0.7 + 0.2 * t
    xs.push([b, b])
    ys.push(1.0)
  }
  let data = match @moongrad.Dataset::new(xs, ys, 1) {
    Some(d) => d
    None => {
      println("| the dataset should be well formed")
      return
    }
  }

  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()

  // Two inputs: the point's coordinates.
  let input = g.input(0)
  b.set_input(
    0,
    match
      @moongrad.Tensor::matrix([
        [0.1, 0.1],
        [0.15, 0.12],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.7, 0.7],
        [0.8, 0.75],
        [0.85, 0.85],
        [0.9, 0.9],
      ]) {
      Some(t) => t
      None => return
    },
  )

  // A hidden layer of eight tanh units.
  let hidden = @moongrad.Linear::new(g, 8, 2, 17)
  let hidden_out = g.unary(@moongrad.UnaryOp::Tanh, hidden.apply(g, input))

  // One output unit with a sigmoid, and a binary cross-entropy
  // loss computed from the logit.
  let output = @moongrad.Linear::new(g, 1, 8, 23)
  let logit = output.apply(g, hidden_out)

  let trainer = @moongrad.TensorTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Adam(0.05),
    fn(g, b) {
      let (loss_node, params) = build_loss(g, b, logit, hidden, output)
      (loss_node, params)
    },
  )

  println("| step     loss        |grad|")
  println("| ------------------------")
  for i = 0; i < 400; i = i + 1 {
    let report = trainer.step()
    if i % 40 == 0 {
      println(
        "| \{i}\t  \{@moongrad.format_double(report.loss)}\t  \{@moongrad.format_double(report.grad_norm)}",
      )
    }
  }
  let final = trainer.final_loss()
  println("| final loss = \{@moongrad.format_double(final)}")
  println("|")
  // What the network predicts on the two clusters.
  println("| predictions")
  let values = @moongrad.t_forward(g, b)
  let logits = values[logit]
  for i = 0; i < 8; i = i + 1 {
    let z = logits.data[i]
    let p = 1.0 / (1.0 + @math.exp(-z))
    let want = if i < 4 { 0.0 } else { 1.0 }
    println(
      "|   sample \{i}: logit = \{@moongrad.format_double(z)}, p = \{@moongrad.format_double(p)}, expected \{@moongrad.format_double(want)}",
    )
  }
  println("|")
  println("| The graph was built once and the gradient compiled once;")
  println("| every step above re-evaluated the compiled gradient at the")
  println("| new weights. That is what emitting a gradient program")
  println("| rather than a tape buys.")
}

///|
/// Build the loss node and collect the parameters to optimise.
fn build_loss(
  g : @moongrad.TGraph,
  b : @moongrad.TBinding,
  logit : Int,
  hidden : @moongrad.Linear,
  output : @moongrad.Linear,
) -> (Int, Array[@moongrad.Tensor]) {
  // Binary cross-entropy from the logit: softplus(z) - z*y, averaged
  // over the batch.
  let labels = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
  let mut acc = g.constant(@moongrad.Tensor::scalar(0.0))
  for i = 0; i < 8; i = i + 1 {
    // One row of the batched logit.
    let z = g.slice(logit, 0, i, 1)
    let y = g.constant(@moongrad.Tensor::vector([labels[i]]))
    let term = g.binary(
      @moongrad.BinaryOp::Sub,
      g.softplus(z),
      g.binary(@moongrad.BinaryOp::Mul, z, y),
    )
    acc = g.binary(@moongrad.BinaryOp::Add, acc, term)
  }
  let loss = g.binary(
    @moongrad.BinaryOp::Div,
    acc,
    g.constant(@moongrad.Tensor::scalar(8.0)),
  )
  let params : Array[@moongrad.Tensor] = []
  for p in hidden.tensors() {
    params.push(p)
  }
  for p in output.tensors() {
    params.push(p)
  }
  ignore(b)
  (loss, params)
}
