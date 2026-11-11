// Mini-batch training.
//
// The graph covers one mini-batch, not the whole dataset. The batch
// is an input, so every step writes the next batch into the binding
// and re-evaluates the compiled gradient there: the graph is
// compiled once for the whole run and the gradient program is
// compiled once for the whole run.
//
// That is the property that makes a mini-batch loop affordable. A
// trainer that rebuilt the graph per batch, or recompiled the
// gradient per step, would spend most of its time on work it had
// already done.
//
// The model is a linear one on two features, and the data is
// generated rather than loaded, so the example runs anywhere.

///|
/// The dataset: two features, and a label that is a linear function
/// of them plus a little noise.
fn make_dataset(n : Int32, seed : Int32) -> (Array[Array[Double]], Array[Double]) {
  let rng = @moongrad.Lcg::new(seed)
  let x : Array[Array[Double]] = []
  let y : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    let a = rng.uniform(-2.0, 2.0)
    let b = rng.uniform(-2.0, 2.0)
    x.push([a, b])
    y.push(1.5 * a - 0.75 * b + 0.25 + rng.uniform(-0.05, 0.05))
  }
  (x, y)
}

///|
/// The linear model: a two-element weight, applied elementwise to
/// the two features.
///
/// The features and the labels are inputs, so the graph is built
/// once and reused for every batch.
fn build(graph : @moongrad.TGraph) -> (Int32, @moongrad.Tensor) {
  let x = graph.variable(0)
  let y = graph.variable(1)
  let w = graph.param(0)
  // The weight is a column, so the product is the dot product of
  // each row of the features with the weight.
  let pred = graph.matmul(x, w)
  let diff = graph.sub(pred, y)
  let squared = graph.mul(diff, diff)
  let per_sample = graph.sum_axis(squared, 1)
  let total = graph.sum_axis(per_sample, 0)
  let loss = graph.div(total, graph.constant(@moongrad.Tensor::filled([], 32.0)))
  ignore(x)
  // The parameter the graph reads through `ParamRead`, and the
  // host-side tensor the optimiser updates. The trainer keeps the
  // two in step.
  (loss, @moongrad.Tensor::zeros([2, 1]))
}

///|
fn main {
  println("| moongrad: mini-batch training")
  println("|")
  println("| The label is 1.5*a - 0.75*b + 0.25, with a little noise.")
  println("| The model is a two-element weight, so the answer is")
  println("| [1.5, -0.75].")
  println("|")
  let (x, y) = make_dataset(200, 2024)
  let batch = 32
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Momentum(0.02, 0.9),
    x,
    y,
    0,
    1,
    batch,
    false,
    2024,
    fn(graph : @moongrad.TGraph, _binding : @moongrad.TBinding) -> (
      Int32,
      Array[@moongrad.Tensor],
    ) {
      let (loss_node, weight) = build(graph)
      (loss_node, [weight])
    },
  )
  println("| epoch    loss        |grad|")
  println("| ---------------------------")
  let steps = trainer.steps_per_epoch()
  for e = 0; e < 30; e = e + 1 {
    // The loss is averaged over the epoch rather than read off the
    // last batch: a single batch is one draw from the data, and the
    // point of the epoch is the average.
    let mut total = 0.0
    let mut worst = 0.0
    for i = 0; i < steps; i = i + 1 {
      let report = trainer.step()
      total = total + report.loss
      if report.grad_norm > worst {
        worst = report.grad_norm
      }
    }
    let epoch = e + 1
    println(
      "| \{@moongrad.format_double(epoch.to_double())}      \{@moongrad.format_double(total / steps.to_double())}    \{@moongrad.format_double(worst)}",
    )
  }
  println("|")
  let w = trainer.params[0]
  println("| the recovered weight: [\{@moongrad.format_double(w.data[0])}, \{@moongrad.format_double(w.data[1])}]")
  println("|")
  println("| The graph was built once and the gradient compiled once.")
  println("| Every step above rewrote the batch into the binding and")
  println("| re-evaluated the compiled gradient there, which is what")
  println("| emitting a gradient program rather than a tape buys.")
}
