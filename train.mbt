// train.mbt
//
// The training loop.
//
// Four things happen per step and they are worth naming, because
// each one is a place a naive implementation goes wrong:
//
//   1. build the graph for the batch
//   2. compile the gradient
//   3. update the host-side parameters
//   4. copy the new parameters back into the binding
//
// Step 4 is the one that is easy to forget. The graph reads
// parameters through `ParamGet`, and the binding is what supplies
// their values; a loop that updates the parameters and forgets to
// refresh the binding will happily train on the same weights
// forever while the loss curve stays perfectly flat. That failure
// looks exactly like a learning-rate problem, which is why the loop
// here makes the refresh explicit and unavoidable.
//
// The graph is built once and reused. Its structure does not depend
// on the parameter *values* — only the leaf nodes do — so rebuilding
// it every step would be pure waste. The compiled gradient is
// re-evaluated instead, which is exactly what `reverse.mbt` was
// designed to allow.

///|
/// One sample: features and expected output.
#derive(Debug)
pub(all) struct Sample {
  x : Array[Double]
  y : Double
}

///|
pub extend Sample with Debug::{to_repr}

///|
/// A dataset with a fixed layout.
#derive(Debug)
pub(all) struct Dataset {
  samples : Array[Sample]
  n_in : Int32
  n_out : Int32
}

///|
pub extend Dataset with Debug::{to_repr}

///|
/// A dataset from parallel arrays. Returns `None` on a ragged
/// input rather than padding, for the same reason `Matrix::from_array`
/// does: a silently reshaped dataset is worse than a rejected one.
pub fn Dataset::new(
  x : Array[Array[Double]],
  y : Array[Double],
  n_out : Int32,
) -> Dataset? {
  if x.length() != y.length() {
    return None
  }
  let n_in = if x.length() > 0 { x[0].length() } else { 0 }
  for row in x {
    if row.length() != n_in {
      return None
    }
  }
  Some({ samples: build_samples(x, y), n_in, n_out, })
}

///|
fn build_samples(x : Array[Array[Double]], y : Array[Double]) -> Array[Sample] {
  let out : Array[Sample] = []
  for i = 0; i < x.length(); i = i + 1 {
    out.push(Sample::{ x: x[i], y: y[i], })
  }
  out
}

///|
/// Shuffle indices with a deterministic generator. Returns indices
/// rather than moving the data, so the caller's arrays stay put.
pub fn shuffled_indices(n : Int32, rng : Lcg) -> Array[Int32] {
  let idx : Array[Int32] = []
  for i = 0; i < n; i = i + 1 {
    idx.push(i)
  }
  // Fisher-Yates, walking down so each swap draws exactly one value.
  for i = n - 1; i > 0; i = i - 1 {
    let j = (rng.next_f64() * (i + 1).to_double()).to_int()
    let t = idx[i]
    idx[i] = idx[j]
    idx[j] = t
  }
  idx
}

///|
/// What one step produced.
#derive(Debug)
pub(all) struct StepReport {
  /// Objective value at this step.
  loss : Double
  /// Euclidean norm of the parameter gradient.
  grad_norm : Double
  /// Whether every number involved stayed finite.
  ok : Bool
}

///|
pub extend StepReport with Debug::{to_repr}

///|
/// A training run, holding everything the loop needs between steps.
#derive(Debug)
pub(all) struct Trainer {
  /// The graph, built once.
  graph : Graph
  /// The network, for its parameter list.
  net : MLP
  /// The dataset.
  data : Dataset
  /// Which loss to use.
  loss_kind : LossKind
  /// Optimiser configuration.
  mut opt_kind : Optimizer
  mut opt_state : OptimizerState
  /// The flat parameter array the optimiser steps.
  flat : Array[Double]
  /// The compiled gradient program.
  grad_graph : Graph
  var_adj : Array[Int32]
  /// The primal slot the loss is rooted at, used to re-evaluate the
  /// forward pass after the parameters move.
  primal : Graph
  /// The node the loss is rooted at.
  out : Int32
  /// The compiled gradient, kept whole so `param_gradients` can
  /// reach the accumulated adjoint circuits.
  compiled : GradGraph
  /// Steps taken.
  mut step_count : Int32
  /// Loss values per step, for the caller's plot.
  mut history : Array[Double]
}

///|
pub extend Trainer with Debug::{to_repr}

///|
/// Which loss the trainer should build.
#derive(Debug, Eq)
pub(all) enum LossKind {
  /// Mean squared error. For regression.
  Mse
  /// Binary cross-entropy. For a single sigmoid output.
  Bce
}

///|
pub extend LossKind with Debug::{to_repr}

///|
pub extend LossKind with Eq::{equal, not_equal}

///|
/// Build a trainer over the whole dataset, as one full batch.
///
/// The graph is built once and reused for every epoch. The features
/// of every sample are variable slots and the weights are parameter
/// slots, so nothing about the structure depends on the numbers —
/// only the leaves move. That is what makes "compile once, evaluate
/// every step" possible at all.
///
/// Full batch rather than minibatch because the alternative would
/// need a graph per batch size, and a graph per step would throw
/// away the whole point. The trade is that the update direction is
/// the exact gradient of the full objective rather than a noisy
/// estimate of it; for the demonstrations here that is a feature,
/// and the examples print the loss curve so the difference is
/// visible.
pub fn Trainer::new(
  sizes : Array[Int32],
  acts : Array[Activation],
  data : Dataset,
  loss_kind : LossKind,
  opt_kind : Optimizer,
  seed : Int32,
) -> Trainer raise ADError {
  let g = Graph::new()
  let n_samples = data.samples.length()
  // One variable per (sample, feature).
  ignore(g.new_vars(n_samples * data.n_in))
  let net = MLP::new(g, sizes, acts, seed)
  let params = net.params()
  let flat : Array[Double] = []
  for p in params {
    let arr = p.to_array()
    for v in arr {
      flat.push(v)
    }
  }
  // Forward pass over the whole dataset, accumulated into one loss.
  let mut acc = g.constant(0.0)
  for s = 0; s < n_samples; s = s + 1 {
    let input : Array[Int32] = []
    for f = 0; f < data.n_in; f = f + 1 {
      input.push(g.variable(s * data.n_in + f))
    }
    let pred = net.apply(g, input)
    let y = data.samples[s].y
    let term = match loss_kind {
      LossKind::Mse => mse_one(g, pred[0], y)
      LossKind::Bce => bce_one(g, pred[0], y)
    }
    acc = g.add(acc, term)
  }
  let out = g.div(acc, g.constant(n_samples.to_double()))
  let b = trainer_binding(g, data, flat, net)
  let values = forward_values(g, b)
  let compiled = compile_gradient(g, values, out)
  let opt_state = OptimizerState::new(opt_kind, flat.length())
  {
    graph: g,
    net,
    data,
    loss_kind,
    opt_kind,
    opt_state,
    flat,
    grad_graph: compiled.graph,
    var_adj: compiled.var_adj,
    primal: g,
    out,
    compiled,
    history: [],
    step_count: 0,
  }
}

///|
/// `(pred - y)^2` for one scalar prediction.
pub fn mse_one(g : Graph, pred : Int32, y : Double) -> Int32 {
  let d = g.sub(pred, g.constant(y))
  g.mul(d, d)
}

///|
/// `softplus(z) - z*y` for one scalar prediction.
pub fn bce_one(g : Graph, z : Int32, y : Double) -> Int32 {
  g.sub(g.softplus(z), g.mul(z, g.constant(y)))
}

///|
/// A binding supplying the current parameter values and the
/// dataset's features.
///
/// The features are variables rather than constants here precisely
/// so the caller can perturb a sample and watch the gradient respond;
/// `trainer_binding` fills them from the dataset, and a caller who
/// wants to look at a single sample can overwrite the slots.
pub fn trainer_binding(
  g : Graph,
  data : Dataset,
  flat : Array[Double],
  net : MLP,
) -> Binding {
  let b = Binding::new()
  let n = data.samples.length() * data.n_in
  for s = 0; s < data.samples.length(); s = s + 1 {
    for f = 0; f < data.n_in; f = f + 1 {
      b.set_var(s * data.n_in + f, data.samples[s].x[f])
    }
  }
  ignore(n)
  let params = net.params()
  let mut offset = 0
  for p in params {
    let k = p.weights.len()
    let slice : Array[Double] = []
    for i = 0; i < k; i = i + 1 {
      slice.push(flat[offset + i])
    }
    b.set_param(p.slot, slice)
    offset = offset + k
  }
  ignore(g)
  b
}

///|
/// One training step: evaluate, differentiate, update.
///
/// Returns the loss and gradient norm so the caller can log or stop
/// early without re-deriving anything.
pub fn Trainer::train_step(
  self : Trainer,
  lr_override : Double,
) -> StepReport raise ADError {
  // 1. Forward pass at the current weights.
  let b = trainer_binding(self.graph, self.data, self.flat, self.net)
  let values = forward_values(self.graph, b)
  let loss_value = values[self.out]
  // 2. Re-evaluate the compiled gradient. The graph structure has
  //    not changed, so recompiling would produce the same program
  //    and measure the same coefficients again.
  let grads : Array[Double] = []
  for i = 0; i < self.var_adj.length(); i = i + 1 {
    grads.push(forward(self.grad_graph, b, self.var_adj[i]).output)
  }
  // Parameter gradients: a partial evaluation of the accumulated
  // adjoint circuit, refreshed against the current binding. The
  // primal values have to travel with it — the partial evaluator
  // reads gradient-graph slots below `primal_size` straight out of
  // them rather than re-evaluating the primal.
  let prog = self.program(values)
  let param_grads = flat_param_gradients(prog, b)
  let gn = grad_norm(param_grads)
  // 3. Update the flat parameter array. The override, when present,
  //    replaces the optimiser entirely rather than just rescaling
  //    it, so a caller can ask for plain SGD on the same trainer
  //    and get exactly that.
  if lr_override >= 0.0 {
    self.opt_state.kind = Optimizer::Sgd(lr_override)
  }
  self.step_count = self.step_count + 1
  match self.opt_state.kind {
    Optimizer::Adam(_) | Optimizer::AdamWithSteps(_, _) =>
      self.opt_state.step_adam(self.flat, param_grads, self.step_count)
    _ => self.opt_state.step(self.flat, param_grads)
  }
  // 4. Write the updated weights back into the parameter matrices.
  //    The optimiser works on `flat`, which is a separate array from
  //    the one the parameters hold, so without this the parameters
  //    would keep reporting the values they were initialised with
  //    while the training run happily converged on the new ones.
  self.sync_params()
  self.history.push(loss_value)
  {
    loss: loss_value,
    grad_norm: gn,
    ok: is_finite(loss_value) && is_finite(gn),
  }
}

///|
/// Copy the flat parameter array back into the parameter matrices.
///
/// The optimiser steps over one flat array because that is the
/// layout a gradient arrives in. The parameters hold matrices,
/// because that is the layout a layer reads. This is the join
/// between the two, and it has to happen after every step.
pub fn Trainer::sync_params(self : Trainer) -> Unit {
  let params = self.net.params()
  let mut offset = 0
  for p in params {
    let n = p.weights.len()
    let slice : Array[Double] = []
    for i = 0; i < n; i = i + 1 {
      slice.push(self.flat[offset + i])
    }
    p.set_values(slice)
    offset = offset + n
  }
}

///|
/// The compiled program, reconstructed from the trainer's fields so
/// the gradient API stays the single source of truth rather than
/// having a second, drifting copy of the parameter layout here.
pub fn Trainer::program(self : Trainer, values : Array[Double]) -> Program {
  Program::{
    primal: self.primal,
    grad: self.compiled,
    primal_values: values,
    output: values[self.out],
  }
}

///|
/// Run `steps` iterations and return the final loss.
pub fn Trainer::fit(self : Trainer, steps : Int32) -> Double raise ADError {
  let mut last = 0.0
  for _i = 0; _i < steps; _i = _i + 1 {
    let report = self.train_step(-1.0)
    if !report.ok {
      break
    }
    last = report.loss
  }
  last
}

///|
/// Loss history, for plotting or for a convergence check.
pub fn Trainer::loss_history(self : Trainer) -> Array[Double] {
  self.history.copy()
}

///|
/// The first loss, for comparing against the last.
pub fn Trainer::initial_loss(self : Trainer) -> Double {
  if self.history.length() == 0 {
    0.0
  } else {
    self.history[0]
  }
}

///|
/// The most recent loss.
pub fn Trainer::final_loss(self : Trainer) -> Double {
  if self.history.length() == 0 {
    0.0
  } else {
    self.history[self.history.length() - 1]
  }
}
