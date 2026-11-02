// optim.mbt
//
// Gradient descent, with momentum, and Adam.
//
// The update rules are a few lines each. What is worth writing down
// is why each one exists and what it costs, because the difference
// between them on a badly conditioned problem is the difference
// between a demo and a result.
//
// SGD:      p -= lr * g
// Momentum: v  = mu * v + g
//           p -= lr * v
// Adam:     m  = b1 * m + (1 - b1) * g
//           v  = b2 * v + (1 - b2) * g^2
//           p -= lr * m_hat / (sqrt(v_hat) + eps)
//
// Plain SGD steps in a direction whose length is set by the
// gradient magnitude, so on a surface with curvature ratio 100 it
// overshoots the narrow direction by two orders of magnitude and
// crawls through the wide one. That is the `newton_beats_gradient_
// descent_on_an_ill_conditioned_quadratic` test, in miniature.
//
// Adam divides by the running standard deviation of the gradient,
// which makes the step length roughly `lr` in every direction. It
// is not obviously the right answer — the update no longer depends
// on the size of the gradient at all — and that is exactly the
// point: it turns a problem about curvature into one about
// per-parameter scale.
//
// The `eps` in Adam is not optional. Dividing by an estimated
// standard deviation of zero gives an infinite step, and the
// gradient of a saturated unit is close enough to zero to trigger
// it. 1e-8 is small enough not to bias ordinary updates and large
// enough to hold.

///|
/// How the parameters are updated from their gradients.
pub(all) enum Optimizer {
  /// p -= lr * g
  Sgd(Double)
  /// SGD with a velocity term.
  Momentum(Double, Double)
  /// Adam with the given learning rate.
  Adam(Double)
  /// Adam with an explicit step count, which it needs to correct
  /// the bias in the moment estimates during the first few steps.
  AdamWithSteps(Double, Int)
} derive(Debug, Eq)

///|
pub extend Optimizer with Debug::{to_repr}

///|
pub extend Optimizer with Eq::{equal, not_equal}

///|
/// Per-parameter optimiser state. Held as parallel arrays keyed by
/// parameter index rather than inside the parameters themselves, so
/// a single optimiser can drive several networks.
pub(all) struct OptimizerState {
  mut kind : Optimizer
  /// Velocity for momentum, first moment for Adam.
  m : Array[Double]
  /// Squared velocity for Adam.
  v : Array[Double]
} derive(Debug)

///|
pub extend OptimizerState with Debug::{to_repr}

///|
/// Fresh state for `count` parameters.
pub fn OptimizerState::new(kind : Optimizer, count : Int) -> OptimizerState {
  let m : Array[Double] = []
  let v : Array[Double] = []
  for i = 0; i < count; i = i + 1 {
    m.push(0.0)
    v.push(0.0)
  }
  { kind, m, v, }
}

///|
/// Learning rate configured in the optimiser.
pub fn Optimizer::lr(self : Optimizer) -> Double {
  match self {
    Sgd(x) => x
    Momentum(x, _) => x
    Adam(x) => x
    AdamWithSteps(x, _) => x
  }
}

///|
/// Apply one update step in place.
///
/// `grads` and `state` are parallel arrays of the same length;
/// `params` is the flat backing store the caller updates. Splitting
/// it this way keeps the optimiser ignorant of the graph, which is
/// the whole reason a compiled gradient is useful here: the gradient
/// arrives as a plain array and the optimiser never has to know it
/// came from a tape.
pub fn OptimizerState::step(
  st : OptimizerState,
  params : Array[Double],
  grads : Array[Double],
) -> Unit {
  let n = if params.length() < grads.length() {
    params.length()
  } else {
    grads.length()
  }
  match st.kind {
    Sgd(lr) =>
      for i = 0; i < n; i = i + 1 {
        params[i] = params[i] - lr * grads[i]
      }
    Momentum(lr, mu) =>
      for i = 0; i < n; i = i + 1 {
        st.m[i] = mu * st.m[i] + grads[i]
        params[i] = params[i] - lr * st.m[i]
      }
    Adam(lr) => apply_adam(st, params, grads, lr, n, 1)
    AdamWithSteps(lr, _) => apply_adam(st, params, grads, lr, n, 1)
  }
}

///|
/// One Adam step.
///
/// `t` is the step index, used for the bias correction. Without it
/// the first update is badly wrong in a consistent direction: both
/// moment estimates start at zero, so `m_hat / sqrt(v_hat)` is
/// `0 / 0` for a parameter whose gradient has been zero so far, and
/// `sign(g)` for one that has not — the parameter takes a full-size
/// step in the wrong direction before it has seen anything.
///

///|
fn apply_adam(
  st : OptimizerState,
  params : Array[Double],
  grads : Array[Double],
  lr : Double,
  n : Int,
  t : Int,
) -> Unit {
  let b1 = 0.9
  let b2 = 0.999
  let eps = 1.0e-8
  let step_t = t.to_double()
  let bias_m = 1.0 - @math.pow(b1, step_t)
  let bias_v = 1.0 - @math.pow(b2, step_t)
  for i = 0; i < n; i = i + 1 {
    let g = grads[i]
    st.m[i] = b1 * st.m[i] + (1.0 - b1) * g
    st.v[i] = b2 * st.v[i] + (1.0 - b2) * g * g
    let m_hat = st.m[i] / bias_m
    let v_hat = st.v[i] / bias_v
    params[i] = params[i] - lr * m_hat / (v_hat.sqrt() + eps)
  }
}

///|
/// Adam, with the bias correction wired to a step counter that the
/// caller advances. This is the form `train.mbt` uses; the plain
/// `Adam` variant exists for callers driving a single step at a
/// time where the correction is negligible.
pub fn OptimizerState::step_adam(
  st : OptimizerState,
  params : Array[Double],
  grads : Array[Double],
  t : Int,
) -> Unit {
  let lr = st.kind.lr()
  let n = if params.length() < grads.length() {
    params.length()
  } else {
    grads.length()
  }
  apply_adam(st, params, grads, lr, n, t)
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

///|
/// Euclidean norm of a gradient. The single most useful number when
/// watching a training run: it falls when the optimiser is making
/// progress and spikes when it is about to diverge.
pub fn grad_norm(grads : Array[Double]) -> Double {
  let mut acc = 0.0
  for g in grads {
    acc = acc + g * g
  }
  acc.sqrt()
}

///|
/// Cosine similarity between two gradients. Useful for spotting a
/// layer that has stopped receiving gradient — in a healthy network
/// the cosine between consecutive gradients stays close to one, and a
/// layer whose cosine collapses is a layer whose ReLUs have all
/// died.
pub fn grad_cosine(a : Array[Double], b : Array[Double]) -> Double {
  let n = if a.length() < b.length() { a.length() } else { b.length() }
  let mut dot = 0.0
  let mut na = 0.0
  let mut nb = 0.0
  for i = 0; i < n; i = i + 1 {
    dot = dot + a[i] * b[i]
    na = na + a[i] * a[i]
    nb = nb + b[i] * b[i]
  }
  if na == 0.0 || nb == 0.0 {
    return 0.0
  }
  dot / (na.sqrt() * nb.sqrt())
}
