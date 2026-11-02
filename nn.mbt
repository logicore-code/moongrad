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
} derive(Debug, Eq)

///|
/// Apply an activation to one scalar node.
pub fn activate(g : Graph, a : Activation, x : Int) -> Int {
  match a {
    Activation::Identity => x
    Activation::Tanh => g.tanh(x)
    Activation::Sigmoid => g.sigmoid(x)
    Activation::Relu => g.relu(x)
  }
}

///|
/// A parameter vector living in its own graph slot.
pub(all) struct Param {
  /// The graph slot, so `Graph::param` can read elements.
  slot : Int
  /// Host-side copy, for the optimiser to update.
  mut weights : Matrix
} derive(Debug)

///|
/// Declare a parameter of `rows x cols` shape in `g` and give it a
/// host-side backing store.
pub fn new_param(g : Graph, rows : Int, cols : Int) -> Param {
  let slot = g.new_param(rows * cols)
  { slot, weights: Matrix::zeros(rows, cols), }
}

///|
/// Read element `(i, j)` of a parameter as a graph node.
pub fn Param::get(
  self : Param,
  g : Graph,
  i : Int,
  j : Int,
) -> Int raise ADError {
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
pub(all) struct Dense {
  w : Param
  b : Param
  has_bias : Bool
  act : Activation
} derive(Debug)

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
  fan_in : Int,
  fan_out : Int,
  act : Activation,
  seed : Int,
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
  x : Array[Int],
) -> Array[Int] raise ADError {
  let out : Array[Int] = []
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
pub(all) struct MLP {
  layers : Array[Dense]
  param_slots : Array[Int]
} derive(Debug)

///|
/// A sequential stack. The last layer's activation is usually
/// `Identity` or `Sigmoid` depending on what the loss expects, so it
/// is set per layer rather than inferred.
pub fn MLP::new(
  g : Graph,
  sizes : Array[Int],
  acts : Array[Activation],
  seed : Int,
) -> MLP {
  let layers : Array[Dense] = []
  let param_slots : Array[Int] = []
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
  x : Array[Int],
) -> Array[Int] raise ADError {
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
pub fn MLP::size(self : MLP) -> Int {
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
pub fn mse_loss(g : Graph, pred : Array[Int], target : Array[Double]) -> Int {
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
pub fn bce_loss(g : Graph, pred : Array[Int], target : Array[Double]) -> Int {
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
pub(all) struct Lcg {
  // Kept in two halves because MoonBit's shift operators are defined
  // on 32-bit integers, and silently truncating a 64-bit state would
  // leave the generator with a period short enough to repeat
  // weights inside a single layer.
  mut hi : Int
  mut lo : Int
} derive(Debug)

///|
/// An LCG with a fixed seed, in the 64-bit constants from Numerical
/// Recipes: x' = 6364136223846793005 * x + 1442695040888963407.
pub fn Lcg::new(seed : Int) -> Lcg {
  { hi: seed, lo: 0, }
}

///|
/// One step of the generator, multiplying the 128-bit product by
/// hand. The high 32 bits of the result are the ones worth keeping:
/// they are the most decorrelated part of the state.
fn Lcg::step(self : Lcg) -> Int {
  let a_hi = 0x5851F42D
  let a_lo = 0x4C957F2D
  // (hi:a_lo : lo) * (a_hi:a_lo)
  let t0 = self.lo * a_lo
  let t1 = self.hi * a_lo + (t0 >> 32)
  let lo = (t1 << 32) | (t0 & 0xFFFFFFFF)
  let hi = self.hi * a_hi + (t1 >> 32) + 0x14057B7E
  self.hi = hi
  self.lo = lo
  hi
}

///|
/// Next value in `[0, 1)`.
pub fn Lcg::next_f64(self : Lcg) -> Double {
  // 52 bits of mantissa, assembled from two 32-bit draws.
  //
  // Assembling the two halves as a single integer does not work:
  // MoonBit's `Int` is 32 bits, so a 53-bit value cannot be held in
  // one at all, and `hi << 27` overflows into the sign bit and
  // makes the generator hand out values just below zero. Scaling
  // the two halves separately keeps every intermediate inside the
  // representable range and every result inside [0, 1).
  let hi = self.step() & 0x001FFFFF // 21 bits, scaled by 2^-21
  let lo = self.step() & 0x7FFFFFFF // 31 bits, scaled by 2^-53
  hi.to_double() / 2097152.0 + lo.to_double() / 9007199254740992.0
}

///|
/// Next value in `[lo, hi)`.
pub fn Lcg::uniform(self : Lcg, lo : Double, hi : Double) -> Double {
  lo + (hi - lo) * self.next_f64()
}
