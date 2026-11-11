// optim_tensor.mbt
//
// Gradient descent over tensors.
//
// The update rules are the same as the scalar optimiser's — SGD,
// momentum, Adam — but the state is per *element* rather than per
// parameter, because a tensor parameter can have thousands of
// entries and a single learning rate for the whole tensor would be
// the wrong granularity.
//
// Adam's bias correction matters here more than in the scalar case.
// The moment estimates start at zero, so without the correction the
// first update is `lr * sign(g)` — a full-size step in a direction
// the optimiser has not earned yet. On a convolutional weight with
// ten thousand entries that is a large step in ten thousand
// directions at once, and it is the difference between a run that
// converges and a run that diverges on the first step.

///|
/// The optimiser's configuration.
#derive(Debug, Eq)
pub(all) enum TensorOptimizer {
  /// p -= lr * g
  Sgd(Double)
  /// SGD with a velocity term.
  Momentum(Double, Double)
  /// Adam with the given learning rate.
  Adam(Double)
}

///|
pub extend TensorOptimizer with Debug::{to_repr}

///|
pub extend TensorOptimizer with Eq::{equal, not_equal}

///|
/// Per-parameter optimiser state.
///
/// The moment estimates live in one flat array over every parameter
/// element, with `layout` giving each parameter's first offset.
/// Flattening them keeps the inner loop a straight walk over
/// contiguous memory, which is where the time actually goes.
#derive(Debug)
pub(all) struct TensorOptState {
  kind : TensorOptimizer
  m : Array[Double]
  v : Array[Double]
  /// The flat offset of each parameter's first element.
  layout : Array[Int32]
}

///|
pub extend TensorOptState with Debug::{to_repr}

///|
/// Fresh state for the given parameter shapes.
///
/// `shapes` gives each parameter's shape, so the state can be sized
/// once and the offsets computed up front.
pub fn TensorOptState::new(
  kind : TensorOptimizer,
  shapes : Array[Array[Int32]],
) -> TensorOptState {
  let m : Array[Double] = []
  let v : Array[Double] = []
  let layout : Array[Int32] = []
  let mut total = 0
  for i = 0; i < shapes.length(); i = i + 1 {
    layout.push(total)
    // The size of a tensor is the product of its shape, not the sum:
    // a 2x3 parameter has six elements, and sizing the state by five
    // would have two parameters share a moment slot.
    let mut count = 1
    for d in shapes[i] {
      count = count * d
    }
    total = total + count
  }
  for i = 0; i < total; i = i + 1 {
    m.push(0.0)
    v.push(0.0)
  }
  { kind, m, v, layout, }
}

///|
/// The learning rate the optimiser is configured with.
pub fn TensorOptimizer::lr(self : TensorOptimizer) -> Double {
  match self {
    Sgd(x) => x
    Momentum(x, _) => x
    Adam(x) => x
  }
}

///|
/// Apply one update step to a list of parameter tensors.
///
/// `params` and `grads` are parallel lists; the caller owns them and
/// sees the updated values in place.
pub fn TensorOptState::step(
  self : TensorOptState,
  params : Array[Tensor],
  grads : Array[Tensor],
  step_index : Int32,
) -> Unit {
  let n = if params.length() < grads.length() {
    params.length()
  } else {
    grads.length()
  }
  match self.kind {
    Sgd(lr) =>
      for i = 0; i < n; i = i + 1 {
        let p = params[i]
        let g = grads[i]
        let k = if p.data.length() < g.data.length() {
          p.data.length()
        } else {
          g.data.length()
        }
        for j = 0; j < k; j = j + 1 {
          p.data[j] = p.data[j] - lr * g.data[j]
        }
        params[i] = p
      }
    Momentum(lr, mu) =>
      for i = 0; i < n; i = i + 1 {
        let p = params[i]
        let g = grads[i]
        let k = if p.data.length() < g.data.length() {
          p.data.length()
        } else {
          g.data.length()
        }
        for j = 0; j < k; j = j + 1 {
          let idx = self.layout[i] + j
          self.m[idx] = mu * self.m[idx] + g.data[j]
          p.data[j] = p.data[j] - lr * self.m[idx]
        }
        params[i] = p
      }
    Adam(lr) => {
      let b1 = 0.9
      let b2 = 0.999
      let eps = 1.0e-8
      let t = step_index.to_double()
      let bias_m = 1.0 - @math.pow(b1, t)
      let bias_v = 1.0 - @math.pow(b2, t)
      for i = 0; i < n; i = i + 1 {
        let p = params[i]
        let g = grads[i]
        let k = if p.data.length() < g.data.length() {
          p.data.length()
        } else {
          g.data.length()
        }
        for j = 0; j < k; j = j + 1 {
          let idx = self.layout[i] + j
          let gv = g.data[j]
          self.m[idx] = b1 * self.m[idx] + (1.0 - b1) * gv
          self.v[idx] = b2 * self.v[idx] + (1.0 - b2) * gv * gv
          let m_hat = self.m[idx] / bias_m
          let v_hat = self.v[idx] / bias_v
          p.data[j] = p.data[j] - lr * m_hat / (v_hat.sqrt() + eps)
        }
        params[i] = p
      }
    }
  }
}
