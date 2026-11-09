// train_tensor.mbt
//
// The tensor training loop.
//
// The same four steps as the scalar loop — build, differentiate,
// update, refresh — with tensors instead of numbers. The important
// difference from the scalar loop is that the graph is built *once*
// and the gradient is *re-evaluated* every step, which is only
// possible because the gradient is a program.
//
// The parameters live in two places at once: as `ParamRead` nodes
// inside the graph, and as host-side tensors the optimiser updates.
// `sync_params` is what keeps them in step, and it is the single
// place a mistake there would show up as a training run that
// converges perfectly and reports its initial weights.

///|
/// One step of the training loop.
#derive(Debug)
pub(all) struct TensorStep {
  /// The objective value at this step.
  loss : Double
  /// The Euclidean norm of the parameter gradient.
  grad_norm : Double
  /// Whether every number involved stayed finite.
  ok : Bool
}

///|
pub extend TensorStep with Debug::{to_repr}

///|
/// A training run over tensors.
#derive(Debug)
pub(all) struct TensorTrainer {
  graph : TGraph
  binding : TBinding
  /// The host-side parameters, which the optimiser updates.
  params : Array[Tensor]
  /// The optimiser's state.
  opt : TensorOptState
  /// The node the loss is rooted at.
  loss_node : Int32
  /// The loss at every step so far.
  history : Array[Double]
  mut step_count : Int32
}

///|
pub extend TensorTrainer with Debug::{to_repr}

///|
/// Build a trainer.
///
/// `build` is given the graph and the binding and has to return the
/// loss node and the parameter tensors it wants optimised. It is a
/// closure because the graph has to be built in a particular order:
/// the inputs first, then the layers, then the loss.
pub fn TensorTrainer::new(
  graph : TGraph,
  binding : TBinding,
  opt_kind : TensorOptimizer,
  build : (TGraph, TBinding) -> (Int32, Array[Tensor]),
) -> TensorTrainer {
  let (loss_node, params) = build(graph, binding)
  // Seed the binding with the parameters the layers just created.
  // The graph reads them through `ParamRead` nodes, and without this
  // the forward pass sees an empty binding and every parameter
  // evaluates to a zero scalar — a failure that looks like a
  // network that simply does not learn.
  for i = 0; i < params.length(); i = i + 1 {
    binding.set_param(i, params[i])
  }
  let shapes : Array[Array[Int32]] = []
  for p in params {
    shapes.push(p.shape.copy())
  }
  let opt = TensorOptState::new(opt_kind, shapes)
  { graph, binding, params, opt, loss_node, history: [], step_count: 0, }
}

///|
/// One training step: evaluate, differentiate, update.
///
/// The forward pass runs on the current parameters, the gradient
/// program is re-evaluated against the same binding, and the
/// optimiser updates the host-side parameter tensors.
pub fn TensorTrainer::step(self : TensorTrainer) -> TensorStep {
  let values = t_forward(self.graph, self.binding)
  let loss_value = values[self.loss_node].sum()
  let grads = t_grad(self.graph, self.binding, self.loss_node)
  let gn = grad_norm_tensors(grads.param_grads)
  self.step_count = self.step_count + 1
  self.opt.step(self.params, grads.param_grads, self.step_count)
  self.sync_params()
  self.history.push(loss_value)
  {
    loss: loss_value,
    grad_norm: gn,
    ok: is_finite(loss_value) && is_finite(gn),
  }
}

///|
/// Run `steps` iterations and return the final loss.
pub fn TensorTrainer::fit(
  self : TensorTrainer,
  steps : Int32,
) -> Double raise ADError {
  let mut last = 0.0
  for i = 0; i < steps; i = i + 1 {
    let report = self.step()
    if !report.ok {
      break
    }
    last = report.loss
  }
  last
}

///|
/// Copy the host-side parameter tensors into the binding.
///
/// This is the step that is easy to forget. The graph reads the
/// parameters through `ParamRead` nodes, and the binding is what
/// supplies their values; a loop that updates the parameters and
/// never refreshes the binding trains forever on stale weights and
/// produces a perfectly flat loss curve — which reads exactly like
/// a learning-rate problem.
pub fn TensorTrainer::sync_params(self : TensorTrainer) -> Unit {
  for i = 0; i < self.params.length(); i = i + 1 {
    self.binding.set_param(i, self.params[i])
  }
}

///|
/// The loss history.
pub fn TensorTrainer::loss_history(self : TensorTrainer) -> Array[Double] {
  self.history.copy()
}

///|
/// The first recorded loss.
pub fn TensorTrainer::initial_loss(self : TensorTrainer) -> Double {
  if self.history.length() == 0 {
    0.0
  } else {
    self.history[0]
  }
}

///|
/// The most recent loss.
pub fn TensorTrainer::final_loss(self : TensorTrainer) -> Double {
  if self.history.length() == 0 {
    0.0
  } else {
    self.history[self.history.length() - 1]
  }
}

///|
/// The Euclidean norm over a list of gradient tensors.
pub fn grad_norm_tensors(grads : Array[Tensor]) -> Double {
  let mut acc = 0.0
  for g in grads {
    for x in g.data {
      acc = acc + x * x
    }
  }
  acc.sqrt()
}

///|
/// The largest absolute entry across a list of gradient tensors.
pub fn grad_max_abs(grads : Array[Tensor]) -> Double {
  let mut best = 0.0
  for g in grads {
    let m = g.max_abs()
    if m > best {
      best = m
    }
  }
  best
}
