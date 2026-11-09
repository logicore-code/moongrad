// nn_tensor.mbt
//
// Neural-network layers over tensors.
//
// Every layer is a small piece of graph construction plus a host-side
// copy of its parameters. The parameters live in the graph as
// `ParamRead` nodes, so the gradient compiler differentiates them
// along with everything else; the host-side copies are what the
// optimiser updates.
//
// The layers here are the ones whose backward rule is either
// expressible with the tensor primitives already in `tensor_op.mbt`
// or worth writing out by hand:
//
//   Linear       y = x @ W + b              — a matmul and an add
//   Conv2d       a 2-D convolution           — im2col and a matmul
//   Pool2d       max or average pooling      — a gather and a scatter
//   BatchNorm    per-channel normalisation  — statistics and a rescale
//   Embedding    a lookup table             — a gather
//   Dropout      a random mask              — a multiply
//
// The activations are the same set as the scalar engine's, but
// applied to whole tensors.

///|
/// A fully-connected layer.
pub(all) struct Linear {
  /// Graph slot of the weight matrix.
  slot_w : Int
  /// Graph slot of the bias vector.
  slot_b : Int
  /// Host-side weights, `(in_features, out_features)`. The layout
  /// is the one `x @ W` needs, not the one a textbook writes down.
  w : Tensor
  /// Host-side bias, `(out_features)`.
  b : Tensor
  in_features : Int
  out_features : Int
} derive(Debug)

///|
/// A dense layer with weights drawn from the Glorot uniform
/// distribution on `[-limit, limit]`, where
/// `limit = sqrt(6 / (fan_in + fan_out))`.
///
/// The scale is not a detail. Too small and the units never leave
/// their linear region; too large and the first forward pass
/// already saturates them. With both fan counts known, the
/// Glorot bound keeps the pre-activation variance near one through a
/// single layer.
pub fn Linear::new(
  g : TGraph,
  out_features : Int,
  in_features : Int,
  seed : Int,
) -> Linear {
  let slot_w = g.param(out_features * in_features)
  let slot_b = g.param(out_features)
  let limit = (6.0 / (in_features + out_features).to_double()).sqrt()
  let rng = Lcg::new(seed)
  let w = Tensor::zeros([in_features, out_features])
  for i = 0; i < w.data.length(); i = i + 1 {
    w.data[i] = rng.uniform(-limit, limit)
  }
  let b = Tensor::zeros([out_features])
  { slot_w, slot_b, w, b, in_features, out_features, }
}

///|
/// `y = x @ W + b`, where `x` is `(batch, in_features)`.
pub fn Linear::apply(self : Linear, g : TGraph, x : Int) -> Int {
  let h = g.matmul(x, g.param(self.slot_w))
  g.binary(BinaryOp::Add, h, g.param(self.slot_b))
}

///|
/// The layer's parameters, as host-side tensors.
pub fn Linear::tensors(self : Linear) -> Array[Tensor] {
  [self.w, self.b]
}

///|
/// Write host-side values back into the parameter slots.
pub fn Linear::write_back(self : Linear, b : TBinding) -> Unit {
  b.set_param(self.slot_w, self.w)
  b.set_param(self.slot_b, self.b)
}

///|
/// An embedding table: a lookup from integer indices to vectors.
pub(all) struct Embedding {
  slot : Int
  table : Tensor
  dim : Int
} derive(Debug)

///|
/// An embedding table with `n` rows of `dim` values, initialised with
/// a small uniform draw. The scale is deliberately small: an
/// embedding starts as an arbitrary vector and the optimiser moves
/// it, so a large initial value just makes the first gradients
/// large.
pub fn Embedding::new(g : TGraph, n : Int, dim : Int, seed : Int) -> Embedding {
  let slot = g.param(n * dim)
  let table = Tensor::zeros([n, dim])
  let rng = Lcg::new(seed)
  let limit = 0.1
  for i = 0; i < table.data.length(); i = i + 1 {
    table.data[i] = rng.uniform(-limit, limit)
  }
  { slot, table, dim, }
}

///|
/// Look up a row of the table. `index` is a flat index into the
/// table, and the result is the row at that index.
pub fn Embedding::row(self : Embedding, g : TGraph, index : Int) -> Int {
  g.gather_row(self.slot, index, self.dim)
}

///|
/// The layer's parameters.
pub fn Embedding::tensors(self : Embedding) -> Array[Tensor] {
  [self.table]
}

///|
/// Write host-side values back into the parameter slot.
pub fn Embedding::write_back(self : Embedding, b : TBinding) -> Unit {
  b.set_param(self.slot, self.table)
}

// ---------------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------------

///|
/// Dropout, applied during training.
///
/// The mask is drawn once per call from a deterministic generator, so
/// a training run is reproducible. Dropout is not differentiable in
/// the usual sense — the gradient is scaled by the mask and divided
/// by the keep probability, which is the standard "straight-through"
/// estimator.
pub(all) struct Dropout {
  rate : Double
  seed : Int
  mut calls : Int
} derive(Debug)

///|
/// A dropout layer that keeps each unit with probability `1 - rate`.
pub fn Dropout::new(rate : Double, seed : Int) -> Dropout {
  { rate, seed, calls: 0, }
}

///|
/// Apply the mask to a tensor.
pub fn Dropout::apply(self : Dropout, g : TGraph, x : Int) -> Int {
  let keep = 1.0 - self.rate
  let rng = Lcg::new(self.seed + self.calls * 7919)
  self.calls = self.calls + 1
  let shape = g.last_shape(x)
  let mask = Tensor::zeros(shape)
  for i = 0; i < mask.data.length(); i = i + 1 {
    mask.data[i] = if rng.next_f64() < keep { 1.0 / keep } else { 0.0 }
  }
  g.scale_tensor(x, mask)
}

// ---------------------------------------------------------------------------
// Sequential
// ---------------------------------------------------------------------------

///|
/// A chain of layers, applied in order.
pub(all) struct Sequential {
  /// The layers, in order. The representation is a list of closures
  /// over the graph so that a layer can hold whatever state it
  /// needs.
  steps : Array[(TGraph, Int) -> Int]
  /// The parameters of every layer, for the optimiser.
  mut tensors : Array[Tensor]
}

///|
/// An empty chain.
pub fn Sequential::new() -> Sequential {
  { steps: [], tensors: [], }
}

///|
/// Append a layer.
pub fn Sequential::push(
  self : Sequential,
  step : (TGraph, Int) -> Int,
  params : Array[Tensor],
) -> Unit {
  self.steps.push(step)
  for t in params {
    self.tensors.push(t)
  }
}

///|
/// Run the chain.
pub fn Sequential::apply(self : Sequential, g : TGraph, x : Int) -> Int {
  let mut acc = x
  for step in self.steps {
    acc = step(g, acc)
  }
  acc
}

///|
/// The chain's parameters, in order.
pub fn Sequential::params(self : Sequential) -> Array[Tensor] {
  self.tensors.copy()
}

// ---------------------------------------------------------------------------
// Helpers on the graph
// ---------------------------------------------------------------------------
