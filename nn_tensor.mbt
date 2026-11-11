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
//   Embedding    a lookup table             — a gather
//   Dropout      a random mask              — a multiply
//
// The normalisation layers are in `nn_norm.mbt`, which writes them
// once for both the layer and the batch variant.
//
// The activations are the same set as the scalar engine's, but
// applied to whole tensors.

///|
/// A fully-connected layer.
#derive(Debug)
pub(all) struct Linear {
  /// Graph slot of the weight matrix.
  slot_w : Int32
  /// Graph slot of the bias vector.
  slot_b : Int32
  /// Host-side weights, `(in_features, out_features)`. The layout
  /// is the one `x @ W` needs, not the one a textbook writes down.
  w : Tensor
  /// Host-side bias, `(out_features)`.
  b : Tensor
  in_features : Int32
  out_features : Int32
}

///|
pub extend Linear with Debug::{to_repr}

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
  out_features : Int32,
  in_features : Int32,
  seed : Int32,
) -> Linear {
  let slot_w = g.new_param_slot()
  let slot_b = g.new_param_slot()
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
pub fn Linear::apply(self : Linear, g : TGraph, x : Int32) -> Int32 {
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
#derive(Debug)
pub(all) struct Embedding {
  slot : Int32
  table : Tensor
  dim : Int32
}

///|
pub extend Embedding with Debug::{to_repr}

///|
/// An embedding table with `n` rows of `dim` values, initialised with
/// a small uniform draw. The scale is deliberately small: an
/// embedding starts as an arbitrary vector and the optimiser moves
/// it, so a large initial value just makes the first gradients
/// large.
pub fn Embedding::new(
  g : TGraph,
  n : Int32,
  dim : Int32,
  seed : Int32,
) -> Embedding {
  let slot = g.new_param_slot()
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
pub fn Embedding::row(self : Embedding, g : TGraph, index : Int32) -> Int32 {
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
/// The mask is an **input** to the graph rather than a constant in
/// it, so it can be redrawn between steps without rebuilding the
/// graph — which is what makes dropout different from a fixed
/// random projection. The training loop calls `refresh` before each
/// step; the forward pass and the gradient program then see the same
/// mask, so the gradient matches the output it belongs to.
///
/// The generator is deterministic, so a training run with a given
/// seed is reproducible. Dropout is not differentiable in the usual
/// sense — the gradient is scaled by the mask and divided by the
/// keep probability, which is the standard "straight-through"
/// estimator.
#derive(Debug)
pub(all) struct Dropout {
  rate : Double
  seed : Int32
  /// The input slot the mask is read from.
  mut slot : Int32
  /// The mask's shape.
  mut shape : Array[Int32]
  /// The mask the layer is currently using.
  mut mask : Tensor
  mut calls : Int32
}

///|
pub extend Dropout with Debug::{to_repr}

///|
/// A dropout layer that keeps each unit with probability `1 - rate`.
pub fn Dropout::new(rate : Double, seed : Int32) -> Dropout {
  { rate, seed, slot: -1, shape: [], mask: Tensor::zeros([]), calls: 0, }
}

///|
/// Apply the mask to a tensor.
///
/// The mask becomes an input of the graph, so that it can be redrawn
/// between steps without rebuilding the graph.
///
/// The mask's shape is given rather than read from the graph: the
/// graph does not know the shape of a tensor until the first forward
/// pass, and the mask has to exist before that.
pub fn Dropout::apply(
  self : Dropout,
  g : TGraph,
  x : Int32,
  shape : Array[Int32],
) -> Int32 {
  self.slot = g.new_variable_slot()
  self.shape = shape.copy()
  self.draw()
  g.mul(x, g.variable(self.slot))
}

///|
/// Draw a mask and keep it.
fn Dropout::draw(self : Dropout) -> Unit {
  let keep = 1.0 - self.rate
  let rng = Lcg::new(self.seed + self.calls * 7919)
  self.calls = self.calls + 1
  let mask = Tensor::zeros(self.shape)
  for i = 0; i < mask.data.length(); i = i + 1 {
    mask.data[i] = if rng.next_f64() < keep { 1.0 / keep } else { 0.0 }
  }
  self.mask = mask
}

///|
/// Write the current mask into the binding.
///
/// The graph reads the mask from an input slot, so the binding has
/// to hold it. Call this once after the layer has been applied.
pub fn Dropout::bind(self : Dropout, b : TBinding) -> Unit {
  if self.slot < 0 {
    return
  }
  b.set_input(self.slot, self.mask)
}

///|
/// Draw the next mask and write it into the binding.
///
/// Calling this between steps is what makes the dropout different
/// from one fixed mask for the whole run. The forward pass and the
/// gradient program both read the mask from the same slot, so they
/// always agree.
pub fn Dropout::refresh(self : Dropout, b : TBinding) -> Unit {
  if self.slot < 0 {
    return
  }
  self.draw()
  self.bind(b)
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
  steps : Array[(TGraph, Int32) -> Int32]
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
  step : (TGraph, Int32) -> Int32,
  params : Array[Tensor],
) -> Unit {
  self.steps.push(step)
  for t in params {
    self.tensors.push(t)
  }
}

///|
/// Run the chain.
pub fn Sequential::apply(self : Sequential, g : TGraph, x : Int32) -> Int32 {
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
