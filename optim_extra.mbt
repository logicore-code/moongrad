// optim_extra.mbt
//
// More tensor optimisers, and the two things that usually travel with
// them: gradient clipping and learning-rate schedules.
//
// The optimisers here keep the state layout of `TensorOptState` in
// `optim_tensor.mbt` — one flat array of moment estimates over every
// parameter element, and a `layout` array giving each parameter the
// offset of its first element — so that changing optimiser does not
// mean rewriting the training loop.
//
// The numbers each variant carries are the learning rate first, then
// the optimiser's own parameter: RMSProp takes the decay of its
// running average, Adagrad and AdamW take epsilon, Nesterov takes
// the momentum. The moment decays b1 = 0.9 and b2 = 0.999 are fixed,
// the same constants Adam uses.
//
// Two conventions are shared by all of them:
//
//   * Epsilon sits outside the square root, as `sqrt(v) + eps`, the
//     same place Adam puts it in `optim_tensor.mbt`. Inside the root,
//     as `sqrt(v + eps)`, a vanishing variance would leave a
//     denominator of `sqrt(eps)` — ten thousand times smaller than
//     eps itself — and the update would be scaled up by that factor.
//   * Only Adam and AdamW correct the bias in their moment estimates,
//     as `m / (1 - b1^t)` and `v / (1 - b2^t)`. The others have
//     little to correct: Adagrad's accumulator is a sum rather than
//     an average, and RMSProp's running average is left uncorrected,
//     which is what makes its first step about three times the
//     learning rate rather than exactly one.

///|
/// The optimiser's configuration.
#derive(Debug, Eq)
pub(all) enum ExtraOptimizer {
  /// RMSProp: v = decay*v + (1-decay)*g^2 ; p -= lr*g/(sqrt(v)+eps)
  RmsProp(Double, Double)
  /// Adagrad: v += g^2 ; p -= lr*g/(sqrt(v)+eps)
  Adagrad(Double, Double)
  /// Nesterov momentum: v = mu*v + g ; p -= lr*(g + mu*v)
  Nesterov(Double, Double)
  /// AdamW: decoupled weight decay. m,v as in Adam, then
  /// p -= lr*(m_hat/(sqrt(v_hat)+eps) + weight_decay*p)
  AdamW(Double, Double, Double)
}

///|
pub extend ExtraOptimizer with Debug::{to_repr}

///|
pub extend ExtraOptimizer with Eq::{equal, not_equal}

///|
/// The learning rate the optimiser is configured with.
pub fn ExtraOptimizer::lr(self : ExtraOptimizer) -> Double {
  match self {
    RmsProp(x, _) => x
    Adagrad(x, _) => x
    Nesterov(x, _) => x
    AdamW(x, _, _) => x
  }
}

///|
/// Per-parameter optimiser state for the extra optimisers.
///
/// The moment estimates live in one flat array over every parameter
/// element, with `layout` giving each parameter's first offset, the
/// same arrangement `TensorOptState` uses. Flattening them keeps the
/// inner loop a straight walk over contiguous memory, which is where
/// the time actually goes.
#derive(Debug)
pub(all) struct ExtraOptState {
  kind : ExtraOptimizer
  m : Array[Double]
  v : Array[Double]
  /// The flat offset of each parameter's first element.
  layout : Array[Int32]
}

///|
pub extend ExtraOptState with Debug::{to_repr}

///|
/// Fresh state for the given parameter shapes.
///
/// `shapes` gives each parameter's shape, so the state can be sized
/// once and the offsets computed up front.
pub fn ExtraOptState::new(
  kind : ExtraOptimizer,
  shapes : Array[Array[Int32]],
) -> ExtraOptState {
  let m : Array[Double] = []
  let v : Array[Double] = []
  let layout : Array[Int32] = []
  let mut total = 0
  for i = 0; i < shapes.length(); i = i + 1 {
    layout.push(total)
    for d in shapes[i] {
      total = total + d
    }
  }
  for i = 0; i < total; i = i + 1 {
    m.push(0.0)
    v.push(0.0)
  }
  { kind, m, v, layout, }
}

///|
/// Apply one update step to a list of parameter tensors.
///
/// `params` and `grads` are parallel lists; the caller owns them and
/// sees the updated values in place. `step_index` counts the steps
/// taken so far, and is what the bias correction in Adam and AdamW
/// needs.
pub fn ExtraOptState::step(
  self : ExtraOptState,
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
    RmsProp(lr, decay) => {
      let eps = 1.0e-8
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
          // The running average of the squared gradient is not bias
          // corrected, so the first step is
          // lr*g/sqrt((1-decay)*g^2) = lr*sign(g)/sqrt(1-decay) —
          // with the usual decay of 0.9 that is a step three times
          // the learning rate.
          self.v[idx] = decay * self.v[idx] + (1.0 - decay) * gv * gv
          p.data[j] = p.data[j] - lr * gv / (self.v[idx].sqrt() + eps)
        }
        params[i] = p
      }
    }
    Adagrad(lr, eps) =>
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
          // The accumulator is a sum, not an average, so there is no
          // bias to correct. The first step is still a full one:
          // sqrt(v) is exactly |g| on the first step, so the update
          // is lr*sign(g).
          self.v[idx] = self.v[idx] + gv * gv
          p.data[j] = p.data[j] - lr * gv / (self.v[idx].sqrt() + eps)
        }
        params[i] = p
      }
    Nesterov(lr, mu) =>
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
          // The velocity holds the gradient as it would be after one
          // more step, which is what makes this Nesterov's method
          // rather than plain momentum: the update looks ahead.
          self.m[idx] = mu * self.m[idx] + gv
          p.data[j] = p.data[j] - lr * (gv + mu * self.m[idx])
        }
        params[i] = p
      }
    AdamW(lr, weight_decay, eps) => {
      let b1 = 0.9
      let b2 = 0.999
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
          let pv = p.data[j]
          self.m[idx] = b1 * self.m[idx] + (1.0 - b1) * gv
          self.v[idx] = b2 * self.v[idx] + (1.0 - b2) * gv * gv
          let m_hat = self.m[idx] / bias_m
          let v_hat = self.v[idx] / bias_v
          // The decay is decoupled: it is added to the step and
          // scaled by the learning rate, rather than added to the
          // gradient where the running standard deviation would
          // scale it as well. A parameter whose gradient is small
          // would otherwise be decayed hardest, since a small
          // gradient also gives a small standard deviation.
          let step = m_hat / (v_hat.sqrt() + eps) + weight_decay * pv
          p.data[j] = pv - lr * step
        }
        params[i] = p
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------

///|
/// Scale a set of gradients so their combined Euclidean norm is at
/// most `max_norm`, and return the norm they had beforehand.
///
/// Clipping is what stops one bad batch from ending a run: the
/// gradient is a sum over the batch, so a single pathological
/// example can make the step far too long in a direction nothing else
/// supports. Scaling by one common factor keeps the direction of the
/// gradient, which clipping the entries one at a time does not.
///
/// The norm from before the clipping is returned because that is the
/// number worth watching: a norm that sits at the limit on every step
/// means the limit, not the learning rate, is what is setting the
/// pace.
pub fn clip_grad_norm(grads : Array[Tensor], max_norm : Double) -> Double {
  let mut acc = 0.0
  for g in grads {
    for x in g.data {
      acc = acc + x * x
    }
  }
  let norm = acc.sqrt()
  // Nothing to scale when there is no gradient, when the limit is
  // negative and there is no such norm to clip to, or when the
  // gradient is already inside the limit. A zero norm has no
  // direction to preserve, and dividing by it would give a NaN.
  if norm == 0.0 || max_norm < 0.0 || norm <= max_norm {
    return norm
  }
  let scale = max_norm / norm
  for i = 0; i < grads.length(); i = i + 1 {
    let g = grads[i]
    for j = 0; j < g.data.length(); j = j + 1 {
      g.data[j] = g.data[j] * scale
    }
    grads[i] = g
  }
  norm
}

///|
/// Clamp every gradient entry to `[-max_value, max_value]`.
///
/// This is the blunt version of `clip_grad_norm`: it catches a single
/// enormous entry and leaves the rest of the gradient alone, but it
/// does not bound the length of the step. It is the right tool when
/// one entry is known to be the problem and the shape of the gradient
/// matters more than its size.
pub fn clip_grad_value(grads : Array[Tensor], max_value : Double) -> Unit {
  for i = 0; i < grads.length(); i = i + 1 {
    let g = grads[i]
    for j = 0; j < g.data.length(); j = j + 1 {
      let x = g.data[j]
      if x > max_value {
        g.data[j] = max_value
      } else if x < -max_value {
        g.data[j] = -max_value
      }
    }
    grads[i] = g
  }
}

// ---------------------------------------------------------------------------
// Learning-rate schedules
// ---------------------------------------------------------------------------

///|
/// How the learning rate changes over the course of a run.
///
/// A schedule is a function of the step index, so the caller asks it
/// for the rate at a step rather than the schedule reaching into the
/// optimiser.
#derive(Debug, Eq)
pub(all) enum LrSchedule {
  /// A constant learning rate.
  Constant(Double)
  /// Multiply by `decay` every `step_size` steps.
  Step(Double, Double, Int32)
  /// Multiply by `decay` at every step.
  Exponential(Double, Double)
  /// Cosine decay from `lr` down to `min_lr` over `period` steps, then flat.
  Cosine(Double, Double, Int32)
}

///|
pub extend LrSchedule with Debug::{to_repr}

///|
pub extend LrSchedule with Eq::{equal, not_equal}

///|
/// The learning rate at the given step.
///
/// The step index counts the updates already taken, so step 0 is the
/// first update and the schedule is still at its starting value
/// there.
pub fn LrSchedule::at(self : LrSchedule, step : Int32) -> Double {
  match self {
    Constant(lr) => lr
    Step(lr, decay, step_size) => {
      // A step size of zero would divide by zero, so treat it as no
      // decay at all rather than as an error.
      if step_size <= 0 {
        return lr
      }
      // Integer division counts how many boundaries have been
      // crossed, so the rate drops on the boundary itself.
      let times = step / step_size
      lr * @math.pow(decay, times.to_double())
    }
    Exponential(lr, decay) => lr * @math.pow(decay, step.to_double())
    Cosine(lr, min_lr, period) => {
      // The progress is clamped at both ends: past the period the
      // rate stays at `min_lr` rather than turning back up, which is
      // what the cosine would do if it were allowed to keep going.
      let mut progress = if period <= 0 {
        1.0
      } else {
        step.to_double() / period.to_double()
      }
      if progress < 0.0 {
        progress = 0.0
      }
      if progress > 1.0 {
        progress = 1.0
      }
      // Half the way down the range at the midpoint of the period,
      // and the two ends of the range at the ends of it.
      min_lr + 0.5 * (lr - min_lr) * (1.0 + @math.cos(@math.PI * progress))
    }
  }
}
