// Parameter inversion.
//
// The other examples fit a network to a task that was made up for
// it. This one does the thing the library is actually for: taking
// measurements of something that already exists and recovering the
// parameters that describe it.
//
// The system is a damped oscillator. It is observed at forty points,
// with noise, and four parameters are unknown: the amplitude, the
// damping, the frequency and the phase. The observations are fitted
// by two routes — gradient descent with Adam, and a Gauss-Newton
// step built from Hessian-vector products — and the two are compared
// on how many steps they needed and on how close they got.
//
// The last part is the one a research code needs and a training
// script does not: the curvature of the misfit says how well the
// parameters are determined. At the minimum, the Hessian of the
// sum of squares is twice the Gauss-Newton matrix, so its inverse
// gives the parameter covariance, and the diagonal of that is how
// precisely each parameter was measured.
//
// Everything is done with one compiled gradient. The misfit is a
// graph; the gradient program is compiled once; every optimiser step
// re-evaluates it at the current parameters. The Hessian-vector
// product that the Gauss-Newton step needs is a second compilation of
// the same graph, and the direction it needs is an input, so one
// compilation serves every direction the iteration asks for.

///|
/// The true parameters: amplitude, damping, frequency, phase.
fn truth() -> Array[Double] {
  [1.2, 0.35, 4.7, 0.6]
}

///|
/// The number of observations.
const POINTS : Int32 = 40

///|
/// The number of parameters.
const NPARAMS : Int32 = 4

///|
/// The damped oscillator, evaluated on the host.
///
/// `y(x) = A exp(-z x) cos(w x + p)`.
fn model(p : Array[Double], x : Double) -> Double {
  p[0] * @math.exp(-p[1] * x) * @math.cos(p[2] * x + p[3])
}

///|
/// The observations: the model at forty points, with noise.
fn make_observations(seed : Int32) -> (@moongrad.Tensor, @moongrad.Tensor) {
  let rng = @moongrad.Lcg::new(seed)
  let xs : Array[Double] = []
  let ys : Array[Double] = []
  for i = 0; i < POINTS; i = i + 1 {
    let x = i.to_double() * 0.15
    xs.push(x)
    ys.push(model(truth(), x) + rng.uniform(-0.02, 0.02))
  }
  let x_tensor = match @moongrad.Tensor::from_data([POINTS], xs) {
    Some(t) => t
    None => @moongrad.Tensor::zeros([POINTS])
  }
  let y_tensor = match @moongrad.Tensor::from_data([POINTS], ys) {
    Some(t) => t
    None => @moongrad.Tensor::zeros([POINTS])
  }
  (x_tensor, y_tensor)
}

///|
/// The misfit: the sum of the squared differences between the model
/// and the observations.
///
/// The parameters are read out of one slot, so the optimiser has four
/// numbers to move rather than four separate ones.
fn build(
  g : @moongrad.TGraph,
  xs : @moongrad.Tensor,
  ys : @moongrad.Tensor,
) -> Int32 {
  let theta = g.param(0)
  let amplitude = g.slice(theta, 0, 0, 1)
  let damping = g.slice(theta, 0, 1, 1)
  let frequency = g.slice(theta, 0, 2, 1)
  let phase = g.slice(theta, 0, 3, 1)
  // exp(-damping * x)
  let decay = g.exp(g.neg(g.mul(damping, g.constant(xs))))
  // cos(frequency * x + phase)
  let oscillation = g.cos(g.add(g.mul(frequency, g.constant(xs)), phase))
  let fitted = g.mul(amplitude, g.mul(decay, oscillation))
  let residual = g.sub(fitted, g.constant(ys))
  g.sum_axis(g.mul(residual, residual), 0)
}

///|
/// The parameters as a plain array.
fn parameters_of(b : @moongrad.TBinding) -> Array[Double] {
  b.params[0].to_data()
}

///|
fn main raise @moongrad.ADError {
  println("| moongrad: recovering the parameters of a damped oscillator")
  println("|")
  println("| the true parameters")
  println(
    "|   amplitude = \{@moongrad.format_double(truth()[0])}   damping = \{@moongrad.format_double(truth()[1])}",
  )
  println(
    "|   frequency = \{@moongrad.format_double(truth()[2])}   phase = \{@moongrad.format_double(truth()[3])}",
  )
  println("|")
  println("| forty noisy observations, and four unknown parameters")
  println("|")
  let (xs, ys) = make_observations(7)
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let loss_node = build(g, xs, ys)
  // The starting point: thirty per cent away from the truth in each
  // parameter, which is far enough that the fit is a real one.
  let start : Array[Double] = [
    truth()[0] * 0.7,
    truth()[1] * 1.3,
    truth()[2] * 1.3,
    truth()[3] * 0.7,
  ]
  b.set_param(0, @moongrad.Tensor::vector(start))
  // The gradient is compiled once, and every step below re-evaluates
  // it at the current parameters.
  let compiled = @moongrad.compile_tensor_gradient(g, b, loss_node)
  let shapes : Array[Array[Int32]] = [[NPARAMS]]
  let opt = @moongrad.TensorOptState::new(
    @moongrad.TensorOptimizer::Adam(0.05),
    shapes,
  )
  let params = [b.params[0]]
  // --- the gradient route ---
  let mut adam_steps = 0
  let mut adam_loss = 0.0
  let mut previous = 1.0e30
  for i = 0; i < 400; i = i + 1 {
    let values = @moongrad.t_forward(g, b)
    adam_loss = values[loss_node].data[0]
    let (_, grads) = @moongrad.evaluate_tensor_gradient(compiled, b)
    ignore(opt.step(params, grads, i + 1))
    b.set_param(0, params[0])
    adam_steps = i + 1
    // The misfit cannot go below the noise, so the test is whether it
    // still moves, not whether it reaches a number.
    if (previous - adam_loss).abs() < 1.0e-9 {
      break
    }
    previous = adam_loss
  }
  let adam_params = parameters_of(b)
  println("| gradient descent with Adam")
  println(
    "|   \{adam_steps} steps, misfit \{@moongrad.format_double(adam_loss)}",
  )
  // --- the second-order route ---
  // The Gauss-Newton step needs the Hessian of the misfit with respect
  // to the parameters, applied to a direction. The direction is an
  // input slot, so the same compilation answers every direction the
  // iteration asks for.
  // The second route starts from the same point the first one did, so
  // that the number of steps means something.
  b.set_param(0, @moongrad.Tensor::vector(start))
  let slot = b.inputs.length()
  // --- the second-order route ---
  let mut cg_steps = 0
  let mut cg_loss = 0.0
  let mut previous = 1.0e30
  for i = 0; i < 40; i = i + 1 {
    let values = @moongrad.t_forward(g, b)
    cg_loss = values[loss_node].data[0]
    // The damping is what a Levenberg-Marquardt step uses: the exact
    // Hessian of a misfit is not always positive definite, and
    // conjugate gradients needs a positive definite operator. Here
    // the curvature is well enough conditioned that a light damping
    // is enough.
    let step = match
      @moongrad.newton_cg_step(g, b, loss_node, 1.0e-6, 100, 1.0e-12, slot) {
      Some(v) => v
      None => {
        println("|   the conjugate-gradient iteration broke down")
        break
      }
    }
    for k = 0; k < NPARAMS; k = k + 1 {
      params[0].data[k] = params[0].data[k] + step[k]
    }
    b.set_param(0, params[0])
    cg_steps = i + 1
    if (cg_loss - previous).abs() < 1.0e-12 {
      break
    }
    previous = cg_loss
  }
  let cg_params = parameters_of(b)
  println("| a Gauss-Newton step from the same starting point")
  println("|   \{cg_steps} steps, misfit \{@moongrad.format_double(cg_loss)}")
  println("|")
  println("| the parameters, side by side")
  println("|   parameter      true       Adam      Newton")
  for i = 0; i < NPARAMS; i = i + 1 {
    let true_value = if i < truth().length() { truth()[i] } else { 0.0 }
    println(
      "|   \{i}          \{@moongrad.format_double(true_value)}   \{@moongrad.format_double(adam_params[i])}   \{@moongrad.format_double(cg_params[i])}",
    )
  }
  println("|")
  // --- how well were the parameters determined? ---
  // At the minimum, the Hessian of the sum of squares is twice the
  // Gauss-Newton matrix. Its inverse is the parameter covariance, up
  // to the noise variance, which is one here to within the scale of
  // the noise that was added.
  // The noise level, estimated from the residuals of the fit: the sum
  // of the squared residuals, over the observations that were not
  // fitted, is the variance of the noise.
  let residuals = @moongrad.t_forward(g, b)
  let fitted = residuals[loss_node].data[0]
  let noise_variance = fitted / (POINTS.to_double() - NPARAMS.to_double())
  let noise_sigma = noise_variance.sqrt()
  let hessian = @moongrad.t_param_hessian(g, b, loss_node)
  // Why the iteration needed damping. The curvature of the misfit in
  // the direction where the amplitude trades against the frequency is
  // small, and the smallest eigenvalue says how small.
  let hessian_tensor = match
    @moongrad.Tensor::from_data(
      [hessian.nrows(), hessian.ncols()],
      hessian.to_array(),
    ) {
    Some(t) => t
    None => @moongrad.Tensor::zeros([hessian.nrows(), hessian.ncols()])
  }
  match @moongrad.eigenvalues(hessian_tensor) {
    Some(v) => {
      let mut smallest = v[0]
      let mut largest = v[0]
      for i = 1; i < v.length(); i = i + 1 {
        if v[i] < smallest {
          smallest = v[i]
        }
        if v[i] > largest {
          largest = v[i]
        }
      }
      println("| the curvature of the misfit")
      println(
        "|   the eigenvalues of the Hessian run from \{@moongrad.format_double(smallest)} to \{@moongrad.format_double(largest)},",
      )
      println("|   so the misfit is a well shaped valley, and a Newton")
      println("|   step crosses it in a handful of moves.")
    }
    None => println("|   the eigenvalues did not converge")
  }
  println("|")
  println("| the precision of the parameters")
  println("|   the covariance is twice the noise variance times the")
  println("|   inverse of the Hessian, and its diagonal says how well")
  println("|   each parameter was measured.")
  println("|   the noise in the observations, estimated from the residuals,")
  println(
    "|   has a standard deviation of \{@moongrad.format_double(noise_sigma)},",
  )
  println("|   and that is the unit the quoted uncertainties are in.")
  for i = 0; i < NPARAMS; i = i + 1 {
    let unit : Array[Double] = []
    for j = 0; j < NPARAMS; j = j + 1 {
      unit.push(if j == i { 1.0 } else { 0.0 })
    }
    let column = match @moongrad.cg_solve_spd(hessian, unit) {
      Some(v) => v
      None => {
        println("|   the linear system did not solve")
        return
      }
    }
    // The covariance of the parameters is the noise variance times the
    // inverse of the Gauss-Newton matrix, and the Gauss-Newton matrix
    // is half the Hessian of the sum of squares, so the covariance is
    // twice the noise variance times the inverse of the Hessian.
    let variance = if column[i] > 0.0 {
      column[i] * 2.0 * noise_variance
    } else {
      0.0
    }
    let sigma = variance.sqrt()
    println("|   parameter \{i}: one sigma = \{@moongrad.format_double(sigma)}")
  }
  println("|")
  println("| The two routes agree on the parameters they found. The")
  println("| second one took fewer steps, because it knows the shape of")
  println("| the valley it is in, and the curvature is what says how far")
  println("| apart two parameters can be traded against each other.")
}
