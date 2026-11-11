// nn.mbt
//
// Layers, assembled out of scalar graph nodes.
//
// A dense layer is three lines of graph construction:
//
//   out[i] = sum_k W[i][k] * x[k] + b[i]
//
// Everything else here is bookkeeping — parameter slots, an
// activation choice, a loss, and a way to walk them in order. The
// differentiation is entirely `reverse.mbt`; nothing in this file
// knows the derivative of anything.
//
// Activations are chosen for numerical behaviour rather than
// convenience. Sigmoid and its derivative both saturate, so a deep
// stack of them produces vanishing gradients that no optimiser
// recovers from; tanh is the usual compromise. Softplus appears as
// the numerically stable form of cross-entropy, where the naive
// `log(sigmoid(-z))` loses all precision for large negative z.

///|
/// What a layer does to its input.
#derive(Debug, Eq)
pub(all) enum Activation {
  /// Linear: no nonlinearity at all. Correct for regression heads.
  Identity
  /// tanh, saturating in both directions.
  Tanh
  /// 1 / (1 + e^-x). Saturates on the negative side.
  Sigmoid
  /// ReLU. Cheap, and unbounded above, which is why it does not
  /// vanish — but its derivative is discontinuous at zero, so the
  /// gradient at exactly 0 is defined as 0 here.
  Relu
}

///|
pub extend Activation with Debug::{to_repr}

///|
pub extend Activation with Eq::{equal, not_equal}

///|
/// Apply an activation to one scalar node.
pub fn activate(g : Graph, a : Activation, x : Int32) -> Int32 {
  match a {
    Activation::Identity => x
    Activation::Tanh => g.tanh(x)
    Activation::Sigmoid => g.sigmoid(x)
    Activation::Relu => g.relu(x)
  }
}

///|
/// A parameter vector living in its own graph slot.
#derive(Debug)
pub(all) struct Param {
  /// The graph slot, so `Graph::param` can read elements.
  slot : Int32
  /// Host-side copy, for the optimiser to update.
  mut weights : Matrix
}

///|
pub extend Param with Debug::{to_repr}

///|
/// Declare a parameter of `rows x cols` shape in `g` and give it a
/// host-side backing store.
pub fn new_param(g : Graph, rows : Int32, cols : Int32) -> Param {
  let slot = g.new_param(rows * cols)
  { slot, weights: Matrix::zeros(rows, cols), }
}

///|
/// Read element `(i, j)` of a parameter as a graph node.
pub fn Param::get(
  self : Param,
  g : Graph,
  i : Int32,
  j : Int32,
) -> Int32 raise ADError {
  g.param(self.slot, i * self.weights.cols + j)
}

///|
/// Flattened copy of the weights, matching the slot's layout.
pub fn Param::to_array(self : Param) -> Array[Double] {
  self.weights.to_array()
}

///|
/// Overwrite the parameter's weights with a flat array.
///
/// This is the join between the two layouts a parameter lives in:
/// the optimiser steps over one flat array because that is the shape
/// a gradient arrives in, and the layer reads a matrix because that
/// is the shape it multiplies. The trainer calls this after every
/// step.
pub fn Param::set_values(self : Param, values : Array[Double]) -> Unit {
  self.weights = Matrix::from_flat(self.weights.rows, self.weights.cols, values)
}

///|
/// A fully-connected layer with an optional bias.
#derive(Debug)
pub(all) struct Dense {
  w : Param
  b : Param
  has_bias : Bool
  act : Activation
}

///|
pub extend Dense with Debug::{to_repr}

///|
/// A dense layer with weights drawn from a uniform distribution on
/// `[-limit, limit]`, the standard Glorot-ish default.
///
/// The scale matters more than it looks. Too small and the sigmoid
/// units never leave their linear region and the model never learns;
/// too large and the first forward pass already saturates them. With
/// fan-in and fan-out both known, `limit = sqrt(6 / (fan_in +
/// fan_out))` keeps the pre-activation variance near 1 through a
/// single layer.
pub fn Dense::new(
  g : Graph,
  fan_in : Int32,
  fan_out : Int32,
  act : Activation,
  seed : Int32,
) -> Dense {
  let w = new_param(g, fan_out, fan_in)
  let limit = (6.0 / (fan_in + fan_out).to_double()).sqrt()
  let rng = Lcg::new(seed)
  for i = 0; i < w.weights.len(); i = i + 1 {
    w.weights.data[i] = rng.uniform(-limit, limit)
  }
  let b = new_param(g, fan_out, 1)
  { w, b, has_bias: true, act, }
}

///|
/// `out[i] = act(sum_k W[i][k] * x[k] + b[i])`, where `x` is a
/// vector of `fan_in` graph nodes and the result is `fan_out` graph
/// nodes.
pub fn Dense::apply(
  self : Dense,
  g : Graph,
  x : Array[Int32],
) -> Array[Int32] raise ADError {
  let out : Array[Int32] = []
  for i = 0; i < self.w.weights.rows; i = i + 1 {
    let mut acc = if self.has_bias {
      self.b.get(g, i, 0)
    } else {
      g.constant(0.0)
    }
    for k = 0; k < self.w.weights.cols; k = k + 1 {
      acc = g.add(acc, g.mul(self.w.get(g, i, k), x[k]))
    }
    out.push(activate(g, self.act, acc))
  }
  out
}

///|
/// Parameters of a layer in the order the optimiser should see them:
/// weights first, then bias.
pub fn Dense::params(self : Dense) -> Array[Param] {
  if self.has_bias {
    [self.w, self.b]
  } else {
    [self.w]
  }
}

///|
/// A stack of dense layers with per-layer activations.
#derive(Debug)
pub(all) struct MLP {
  layers : Array[Dense]
  param_slots : Array[Int32]
}

///|
pub extend MLP with Debug::{to_repr}

///|
/// A sequential stack. The last layer's activation is usually
/// `Identity` or `Sigmoid` depending on what the loss expects, so it
/// is set per layer rather than inferred.
pub fn MLP::new(
  g : Graph,
  sizes : Array[Int32],
  acts : Array[Activation],
  seed : Int32,
) -> MLP {
  let layers : Array[Dense] = []
  let param_slots : Array[Int32] = []
  for i = 1; i < sizes.length(); i = i + 1 {
    let layer = Dense::new(
      g,
      sizes[i - 1],
      sizes[i],
      acts[i - 1],
      seed + i * 7919,
    )
    for p in layer.params() {
      param_slots.push(p.slot)
    }
    layers.push(layer)
  }
  { layers, param_slots, }
}

///|
/// Forward pass: run every layer in order.
pub fn MLP::apply(
  self : MLP,
  g : Graph,
  x : Array[Int32],
) -> Array[Int32] raise ADError {
  let mut acc = x
  for layer in self.layers {
    acc = layer.apply(g, acc)
  }
  acc
}

///|
/// Every parameter in the network, weights before biases, layer by
/// layer. The optimiser steps over this list.
pub fn MLP::params(self : MLP) -> Array[Param] {
  let out : Array[Param] = []
  for layer in self.layers {
    for p in layer.params() {
      out.push(p)
    }
  }
  out
}

///|
/// Total number of scalars the network is carrying.
pub fn MLP::size(self : MLP) -> Int32 {
  let mut n = 0
  for layer in self.layers {
    for p in layer.params() {
      n = n + p.weights.len()
    }
  }
  n
}

// ---------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------

///|
/// Mean squared error between a prediction and a target.
///
/// Divided by the batch count rather than by `2 * batch`, so the
/// reported number is the actual average squared error and can be
/// compared against a variance.
pub fn mse_loss(
  g : Graph,
  pred : Array[Int32],
  target : Array[Double],
) -> Int32 {
  let mut acc = g.constant(0.0)
  for i = 0; i < pred.length(); i = i + 1 {
    let d = g.sub(pred[i], g.constant(target[i]))
    acc = g.add(acc, g.mul(d, d))
  }
  let n = pred.length().to_double()
  g.div(acc, g.constant(n))
}

///|
/// Binary cross-entropy written as `softplus(z) - z*y`.
///
/// The textbook form `-y log(sigmoid(z)) - (1-y) log(sigmoid(-z))`
/// computes `log` of something that underflows to zero once `|z|`
/// exceeds about 40, and then produces `-inf` for a perfectly
/// confident correct prediction. This form has no such cliff: the
/// softplus primitive is evaluated as `max(z,0) + log(1 + e^-|z|)`,
/// which stays finite everywhere.
pub fn bce_loss(
  g : Graph,
  pred : Array[Int32],
  target : Array[Double],
) -> Int32 {
  let mut acc = g.constant(0.0)
  for i = 0; i < pred.length(); i = i + 1 {
    let y = target[i]
    let z = pred[i]
    // softplus(z) - z*y
    acc = g.add(acc, g.sub(g.softplus(z), g.mul(z, g.constant(y))))
  }
  let n = pred.length().to_double()
  g.div(acc, g.constant(n))
}

///|
/// The numerically stable cross-entropy, evaluated on the host. Used
/// for reporting, so the printed loss is the value the loss function
/// means and not a quantity the graph had to be simplified into.
pub fn bce_value(z : Double, y : Double) -> Double {
  // log(1 + e^-|z|) + max(z, 0) - z*y, with the log term guarded.
  let log_term = @math.ln(1.0 + @math.exp(-absf(z)))
  maxf(z, 0.0) + log_term - z * y
}

///|
/// The same cross-entropy, averaged, on the host.
pub fn bce_value_mean(pred : Array[Double], target : Array[Double]) -> Double {
  if pred.length() == 0 {
    return 0.0
  }
  let mut acc = 0.0
  for i = 0; i < pred.length(); i = i + 1 {
    acc = acc + bce_value(pred[i], target[i])
  }
  acc / pred.length().to_double()
}

// ---------------------------------------------------------------------------
// Deterministic randomness
// ---------------------------------------------------------------------------

///|
/// A 64-bit linear congruential generator.
///
/// Present because a training run that depends on the platform's
/// random number generator is not reproducible, and a gradient engine
/// whose examples give different answers on different machines is
/// much harder to trust. Numerical Recipes' constants; good enough
/// for weight initialisation, and identical on every target.
#derive(Debug)
pub(all) struct Lcg {
  // The whole 64-bit state. It used to be kept in two 32-bit halves
  // and multiplied by hand, which does not work: MoonBit's `Int` is
  // 32 bits, so the carries a 64-bit product needs cannot be
  // recovered from it, and the generator degenerated into something
  // that produced a narrow band of values rather than a spread one.
  mut state : Int64
}

///|
pub extend Lcg with Debug::{to_repr}

///|
/// An LCG with a fixed seed, in the 64-bit constants from Numerical
/// Recipes: x' = 6364136223846793005 * x + 1442695040888963407.
pub fn Lcg::new(seed : Int32) -> Lcg {
  { state: seed.to_int64(), }
}

///|
/// One step of the generator, multiplying the 128-bit product by
/// hand. The high 32 bits of the result are the ones worth keeping:
/// they are the most decorrelated part of the state.
fn Lcg::step(self : Lcg) -> Int64 {
  // x' = 6364136223846793005 * x + 1442695040888963407, the
  // constants from Numerical Recipes. The multiplication wraps, which
  // is what makes it a generator rather than an accumulator.
  self.state = self.state * 6364136223846793005L + 1442695040888963407L
  self.state
}

///|
/// Next value in `[0, 1)`.
pub fn Lcg::next_f64(self : Lcg) -> Double {
  // The high bits of a linear congruential generator are the good
  // ones and the low bits are the poor ones, so the value is taken
  // from the top of the state: 53 bits, which is exactly what a
  // double's mantissa holds.
  //
  // The shift is arithmetic, so the sign has to be masked off before
  // the bits are used.
  let bits = (self.step() >> 11) & 0x1FFFFFFFFFFFFFL
  bits.to_double() / 9007199254740992.0
}

///|
/// Next value in `[lo, hi)`.
pub fn Lcg::uniform(self : Lcg, lo : Double, hi : Double) -> Double {
  lo + (hi - lo) * self.next_f64()
}
