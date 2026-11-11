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
//
// The gradient program is compiled once, when the trainer is built,
// and re-evaluated at every step. That is only possible because the
// gradient is a program rather than a tape: re-evaluating it at the
// new weights costs one scan of the graph, where re-deriving it
// costs a full backward sweep. On the scalar side the difference is
// about twenty to one, so a trainer that recompiled every step would
// spend nearly all of its time on work it had already done.

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
pub(all) struct TensorTrainer {
  graph : TGraph
  binding : TBinding
  /// The gradient program, compiled once.
  ///
  /// It is compiled on the first step rather than when the trainer
  /// is built, because the backward rules bake the operand shapes
  /// into the program and the shapes come from the binding — which
  /// the caller may not have filled in yet. Compiling later removes
  /// the ordering requirement: the trainer works whether the inputs
  /// are set before or after it is built.
  mut grad : TGradGraph?
  /// The host-side parameters, which the optimiser updates.
  params : Array[Tensor]
  /// The optimiser's state.
  opt : TensorOptState
  /// Callbacks that redraw a layer's stochastic inputs.
  ///
  /// The dropout layer's mask is an input rather than a constant, so
  /// it has to be redrawn between steps for the layer to be a dropout
  /// rather than a fixed random projection. The callback is given
  /// the binding to write into.
  mut refreshers : Array[(TBinding) -> Unit]
  /// The node the loss is rooted at.
  loss_node : Int32
  /// The loss at every step so far.
  history : Array[Double]
  mut step_count : Int32
}

///|
/// Build a trainer.
///
/// `build` is given the graph and the binding and has to return the
/// loss node and the parameter tensors it wants optimised. It is a
/// closure because the graph has to be built in a particular order:
/// the inputs first, then the layers, then the loss.
///
/// The parameters have to come back in **slot order** — the order
/// `new_param_slot` handed the slots out, which is the order the
/// layers were created. A parameter bound to the wrong slot makes
/// the forward pass fail, and the failure looks like a network that
/// is not learning rather than like a mistake; the step reports it
/// as a failure rather than as a zero gradient.
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
  {
    graph,
    binding,
    grad: None,
    params,
    opt,
    refreshers: [],
    loss_node,
    history: [],
    step_count: 0,
  }
}

///|
/// Register a callback that redraws a layer's stochastic inputs
/// before every step.
///
/// A dropout layer is one: its mask lives in an input slot, so
/// `dropout.refresh` has to be called between steps for the layer to
/// drop anything at all.
pub fn TensorTrainer::add_refresher(
  self : TensorTrainer,
  f : (TBinding) -> Unit,
) -> Unit {
  self.refreshers.push(f)
}

///|
/// One training step: evaluate, differentiate, update.
///
/// The forward pass runs on the current parameters, the gradient
/// program is re-evaluated against the same binding, and the
/// optimiser updates the host-side parameter tensors.
pub fn TensorTrainer::step(self : TensorTrainer) -> TensorStep {
  for f in self.refreshers {
    f(self.binding)
  }
  if self.grad is None {
    // The graph's structure does not change during training, so
    // compiling it once is enough; recompiling it every step would
    // redo the same work and measure the same coefficients again.
    self.grad = Some(
      compile_tensor_gradient(self.graph, self.binding, self.loss_node),
    )
  }
  let grad = match self.grad {
    Some(g) => g
    None => return { loss: 0.0, grad_norm: 0.0, ok: false, }
  }
  // The program was compiled for the shapes it was compiled against.
  // If the binding has moved on — a different batch size, a layer
  // rebuilt — the program is not valid here, and evaluating it would
  // quietly produce the gradient of a different function. That is
  // checked before anything is computed.
  if !grad.shapes_match(self.binding) {
    return { loss: 0.0, grad_norm: 0.0, ok: false, }
  }
  // One evaluation of the gradient graph gives the primal's values
  // and the gradient's values together: the gradient program holds
  // the primal below it, and evaluating it evaluates both.
  let values = t_forward(grad.graph, self.binding)
  if grad.graph.failed {
    // The evaluation itself failed: a shape that does not fit, or a
    // parameter that does not match the slot it was bound to. A
    // failure here used to look like a network that simply was not
    // learning.
    return { loss: 0.0, grad_norm: 0.0, ok: false, }
  }
  let loss_value = values[self.loss_node].sum()
  let grads : Array[Tensor] = []
  for i = 0; i < grad.param_adj.length(); i = i + 1 {
    grads.push(values[grad.param_adj[i]])
  }
  let gn = grad_norm_tensors(grads)
  self.step_count = self.step_count + 1
  self.opt.step(self.params, grads, self.step_count)
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
pub fn TensorTrainer::fit(self : TensorTrainer, steps : Int32) -> Double {
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
