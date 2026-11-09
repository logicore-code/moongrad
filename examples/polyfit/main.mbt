// Polynomial regression.
//
// Fits a quadratic to noisy samples of y = 1.5x^2 - 0.8x + 0.25 by
// gradient descent on the mean squared error, then prints the
// recovered coefficients next to the true ones.
//
// The features are [1, x, x^2] and the model is a single linear
// layer over them, so the fitted weights *are* the polynomial's
// coefficients. That makes the example a direct test of the
// parameter plumbing — the part that turns a gradient into an
// update — because a mis-wired bias or a transposed weight matrix
// shows up immediately as a wrong coefficient rather than as a
// vaguely worse loss.
//
// The last line evaluates the fitted model at a point the training
// never saw, which is the only number here that says anything about
// whether the fit generalised rather than merely interpolated.

///|
let true_coeffs : Array[Double] = [0.25, -0.8, 1.5]

///|
fn features(x : Double, degree : Int32) -> Array[Double] {
  let out : Array[Double] = []
  let mut power = 1.0
  for _i = 0; _i <= degree; _i = _i + 1 {
    out.push(power)
    power = power * x
  }
  out
}

///|
fn main raise @moongrad.ADError {
  // Samples with a deterministic pseudo-random perturbation, so the
  // run is reproducible on every machine.
  let rng = @moongrad.Lcg::new(2026)
  let xs : Array[Array[Double]] = []
  let ys : Array[Double] = []
  for i = 0; i < 40; i = i + 1 {
    let x = -2.0 + 4.0 * (i.to_double() / 39.0)
    let noise = (rng.next_f64() - 0.5) * 0.1
    xs.push(features(x, 2))
    ys.push(
      true_coeffs[2] * x * x + true_coeffs[1] * x + true_coeffs[0] + noise,
    )
  }
  let data = match @moongrad.Dataset::new(xs, ys, 1) {
    Some(d) => d
    None => {
      println("| the dataset is malformed")
      return
    }
  }
  // Three features in, one out, no nonlinearity: the model is
  // linear in the features, so the least-squares solution is the
  // exact polynomial.
  let trainer = @moongrad.Trainer::new(
    [3, 1],
    [@moongrad.Activation::Identity],
    data,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    7,
  )
  println("| moongrad polynomial regression")
  println("| fitting y = 1.5x^2 - 0.8x + 0.25 on 40 samples of x in [-2, 2]")
  println("|")
  let first = trainer.train_step(-1.0)
  println("| step 0    loss = \{@moongrad.format_double(first.loss)}")
  for i = 1; i <= 3000; i = i + 1 {
    let r = trainer.train_step(-1.0)
    if i % 500 == 0 {
      println(
        "| step \{i}   loss = \{@moongrad.format_double(r.loss)}   |grad| = \{@moongrad.format_double(r.grad_norm)}",
      )
    }
  }
  println("| final loss = \{@moongrad.format_double(trainer.final_loss())}")
  println("|")
  // The trained weights are the polynomial's coefficients. The layer
  // is 1x3 over [1, x, x^2], so weight[0][k] is the coefficient of
  // the k-th feature.
  let params = trainer.net.params()
  let w = params[0].weights
  let b = params[1].weights
  let names = ["1    ", "x    ", "x^2  "]
  println("| coefficient    true       fitted")
  for k = 0; k < 3; k = k + 1 {
    println(
      "| \{names[k]}  \{@moongrad.format_double(true_coeffs[k])}      \{@moongrad.format_double(w.get(0, k))}",
    )
  }
  println(
    "| bias                           \{@moongrad.format_double(b.get(0, 0))}",
  )
  // Held-out check: evaluate the fitted model at a point the
  // training never saw.
  let probe = 2.5
  let f = features(probe, 2)
  let mut predicted = b.get(0, 0)
  let mut true_value = 0.0
  for k = 0; k < 3; k = k + 1 {
    predicted = predicted + w.get(0, k) * f[k]
    true_value = true_value + true_coeffs[k] * f[k]
  }
  println("|")
  println(
    "| at x = \{probe} (outside the training range): true = \{@moongrad.format_double(true_value)}, fitted = \{@moongrad.format_double(predicted)}",
  )
}
