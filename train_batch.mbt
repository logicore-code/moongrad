// train_batch.mbt
//
// Mini-batch training.
//
// The graph covers one mini-batch, not the whole dataset. That is the
// only arrangement that works with a statically-shaped tensor graph:
// the batch axis has a length, and that length is part of the graph's
// shapes. A graph per batch size would throw away the compile-once
// property, and a graph per step would throw away the compile-once
// property twice over.
//
// The batch is an input. At every step the trainer writes the next
// mini-batch into the binding and re-evaluates the compiled gradient
// there, so the expensive part — deriving the gradient program — is
// paid once for the whole run, not once per step. That is the same
// property that makes the tensor trainer cheap, and it is what makes
// a mini-batch loop affordable at all.
//
// The cost of the arrangement is that the number of samples in a
// batch is fixed when the graph is built. Changing it means a new
// graph, and therefore a new compilation. For a training run the
// batch size does not change, so this costs nothing.

///|
/// One step of a mini-batch training loop.
#derive(Debug)
pub(all) struct BatchStep {
  /// The objective value at this step.
  loss : Double
  /// The Euclidean norm of the parameter gradient.
  grad_norm : Double
  /// The number of samples in the batch that was used.
  samples : Int32
  /// Whether every number involved stayed finite.
  ok : Bool
}

///|
pub extend BatchStep with Debug::{to_repr}

///|
/// A mini-batch training run over tensors.
#derive(Debug)
pub(all) struct BatchTrainer {
  graph : TGraph
  binding : TBinding
  /// The gradient program, compiled once.
  mut grad : TGradGraph
  /// The host-side parameters, which the optimiser updates.
  params : Array[Tensor]
  /// The optimiser's state.
  opt : TensorOptState
  /// The node the loss is rooted at.
  loss_node : Int32
  /// The features, one row per sample.
  features : Array[Array[Double]]
  /// The labels, one per sample.
  labels : Array[Double]
  /// The input slot the features are written to.
  feature_slot : Int32
  /// The input slot the labels are written to.
  label_slot : Int32
  /// How many rows the graph expects in a batch.
  batch_size : Int32
  /// The order the samples are visited in.
  order : Array[Int32]
  /// The flattened feature buffer the batch is written into.
  batch_features : Array[Double]
  /// The flattened label buffer the batch is written into.
  batch_labels : Array[Double]
  /// Where in the order we are.
  mut cursor : Int32
  /// How many times the visit order has been exhausted.
  ///
  /// A pass over the data is one epoch when the batch is smaller
  /// than the dataset; a batch larger than the dataset makes
  /// several passes in one step.
  mut epoch : Int32
  /// The loss at every step so far.
  history : Array[Double]
  mut step_count : Int32
  /// The seed the shuffle is driven by.
  mut seed : Int32
  /// Whether a short final batch is dropped or cycled over.
  drop_last : Bool
}

///|
pub extend BatchTrainer with Debug::{to_repr}

///|
/// Build a mini-batch trainer.
///
/// `build` is given the graph and the binding and returns the loss
/// node and the parameter tensors to optimise. It is a closure
/// because the graph has to be built in a particular order: the
/// inputs first, then the layers, then the loss.
///
/// The features go into input slot `feature_slot` as a
/// `[batch_size, features]` tensor and the labels into `label_slot` as
/// a `[batch_size, 1]` tensor, so that a label broadcasts against a
/// row of predictions. The graph is expected to read them in that
/// shape.
pub fn BatchTrainer::new(
  graph : TGraph,
  binding : TBinding,
  opt_kind : TensorOptimizer,
  features : Array[Array[Double]],
  labels : Array[Double],
  feature_slot : Int32,
  label_slot : Int32,
  batch_size : Int32,
  drop_last : Bool,
  seed : Int32,
  build : (TGraph, TBinding) -> (Int32, Array[Tensor]),
) -> BatchTrainer {
  let (loss_node, params) = build(graph, binding)
  for i = 0; i < params.length(); i = i + 1 {
    binding.set_param(i, params[i])
  }
  let shapes : Array[Array[Int32]] = []
  for p in params {
    shapes.push(p.shape.copy())
  }
  let opt = TensorOptState::new(opt_kind, shapes)
  let width = if features.length() > 0 { features[0].length() } else { 0 }
  let batch_features : Array[Double] = []
  for i = 0; i < batch_size * width; i = i + 1 {
    batch_features.push(0.0)
  }
  let batch_labels : Array[Double] = []
  for i = 0; i < batch_size; i = i + 1 {
    batch_labels.push(0.0)
  }
  let order : Array[Int32] = []
  for i = 0; i < features.length(); i = i + 1 {
    order.push(i)
  }
  let trainer = {
    graph,
    binding,
    // A placeholder: the real gradient is compiled below, once the
    // binding holds a batch. The backward rules bake the operand
    // shapes into the program, and the shapes come from the values
    // at compile time — so compiling against an empty binding would
    // bake in the wrong ones.
    grad: compile_tensor_gradient(graph, binding, -1),
    params,
    opt,
    loss_node,
    features,
    labels,
    batch_features,
    batch_labels,
    feature_slot,
    label_slot,
    batch_size,
    order,
    cursor: 0,
    epoch: 0,
    history: [],
    step_count: 0,
    seed,
    drop_last,
  }
  trainer.shuffle(seed)
  // The backward rules bake the operand shapes into the program, so
  // the binding has to hold a real batch before the gradient is
  // compiled. The fill is undone afterwards: it was there to give the
  // compiler the right shapes, not to consume a batch.
  let cursor = trainer.cursor
  ignore(trainer.fill_batch())
  trainer.cursor = cursor
  trainer.grad = compile_tensor_gradient(graph, binding, loss_node)
  trainer
}

///|
/// The number of samples in the dataset.
pub fn BatchTrainer::samples(self : BatchTrainer) -> Int32 {
  self.features.length()
}

///|
/// Shuffle the visit order.
///
/// The generator is a small linear congruential one, so the same seed
/// always gives the same order. A training run that cannot be
/// reproduced is a training run whose results cannot be checked.
pub fn BatchTrainer::shuffle(self : BatchTrainer, seed : Int32) -> Unit {
  self.seed = seed
  let order = shuffled_indices(self.features.length(), Lcg::new(seed))
  for i = 0; i < order.length(); i = i + 1 {
    self.order[i] = order[i]
  }
}

///|
/// Fill the binding with the next mini-batch.
///
/// Returns the number of samples that went into the batch. A batch
/// that runs off the end of the dataset either cycles back to the
/// start or is dropped, depending on `drop_last`.
///
/// A dropped batch is a short one: fewer rows than the graph was
/// built for. The binding is not written at all in that case, so a
/// step that gets one is skipped rather than run on the previous
/// batch's rows.
pub fn BatchTrainer::fill_batch(self : BatchTrainer) -> Int32 {
  let n = self.order.length()
  let want = self.batch_size
  if n == 0 || want <= 0 {
    // Nothing to draw from, or a batch of nothing.
    return 0
  }
  let width = self.features[0].length()
  let mut taken = 0
  let mut cursor = self.cursor
  while taken < want {
    if cursor >= n {
      if self.drop_last {
        // The short final batch is dropped. The cursor stays where it
        // is, so the next epoch starts from the top of the order.
        break
      }
      cursor = 0
      self.epoch = self.epoch + 1
      self.shuffle(self.seed + self.epoch)
    }
    let idx = self.order[cursor]
    cursor = cursor + 1
    let row = self.features[idx]
    // Every row has to be the same width: the batch is one tensor,
    // and a row of a different width would land at the wrong
    // offsets.
    if row.length() != width {
      return 0
    }
    for j = 0; j < row.length(); j = j + 1 {
      let at = taken * width + j
      self.batch_features[at] = row[j]
    }
    self.batch_labels[taken] = self.labels[idx]
    taken = taken + 1
  }
  self.cursor = cursor
  if taken < want {
    // The batch is short, so it does not fit the graph. Leave the
    // binding alone rather than filling it with the previous batch's
    // rows.
    return taken
  }
  // The features go in as a [batch, features] tensor and the labels
  // as a [batch] vector, which is the shape the graph was built for.
  let ft = Tensor::zeros([self.batch_size, width])
  for i = 0; i < ft.data.length(); i = i + 1 {
    ft.data[i] = self.batch_features[i]
  }
  self.binding.set_input(self.feature_slot, ft)
  // The labels go in as a [batch, 1] tensor, so that they broadcast
  // against a [batch, features] prediction.
  let lt = Tensor::zeros([self.batch_size, 1])
  for i = 0; i < self.batch_size; i = i + 1 {
    lt.data[i] = self.batch_labels[i]
  }
  self.binding.set_input(self.label_slot, lt)
  taken
}

///|
/// One training step: fill the batch, evaluate, update.
///
/// The forward pass runs on the current parameters, the gradient
/// program is re-evaluated against the same binding, and the
/// optimiser updates the host-side parameter tensors.
pub fn BatchTrainer::step(self : BatchTrainer) -> BatchStep {
  let used = self.fill_batch()
  if used < self.batch_size {
    // The batch was short — the last one of the epoch, dropped on
    // purpose. Running the step anyway would train on the previous
    // batch's rows, which is worse than not stepping at all.
    return { loss: 0.0, grad_norm: 0.0, samples: used, ok: false, }
  }
  if !self.grad.shapes_match(self.binding) {
    // The batch changed shape, which the compiled program was not
    // compiled for. Saying so beats returning the gradient of a
    // different function.
    return { loss: 0.0, grad_norm: 0.0, samples: used, ok: false, }
  }
  let values = t_forward(self.grad.graph, self.binding)
  let loss_value = values[self.loss_node].sum()
  let grads : Array[Tensor] = []
  for i = 0; i < self.grad.param_adj.length(); i = i + 1 {
    grads.push(values[self.grad.param_adj[i]])
  }
  let gn = grad_norm_tensors(grads)
  self.step_count = self.step_count + 1
  self.opt.step(self.params, grads, self.step_count)
  self.sync_params()
  self.history.push(loss_value)
  {
    loss: loss_value,
    grad_norm: gn,
    samples: used,
    ok: is_finite(loss_value) && is_finite(gn),
  }
}

///|
/// Run whole epochs and return the final loss.
pub fn BatchTrainer::fit(self : BatchTrainer, epochs : Int32) -> Double {
  let per_epoch = self.steps_per_epoch()
  let mut last = 0.0
  for e = 0; e < epochs; e = e + 1 {
    for i = 0; i < per_epoch; i = i + 1 {
      let report = self.step()
      if !report.ok {
        return report.loss
      }
      last = report.loss
    }
  }
  last
}

///|
/// How many steps an epoch takes.
pub fn BatchTrainer::steps_per_epoch(self : BatchTrainer) -> Int32 {
  let n = self.samples()
  if n <= 0 || self.batch_size <= 0 {
    return 0
  }
  if self.drop_last {
    // A short final batch is thrown away, so the epoch is the
    // number of whole batches.
    let whole = n / self.batch_size
    if whole <= 0 {
      return 1
    }
    whole
  } else {
    // The short final batch is kept, so the epoch rounds up.
    (n + self.batch_size - 1) / self.batch_size
  }
}

///|
/// The mean loss over the steps of the epoch that just finished.
///
/// Only the last epoch's steps are counted: averaging the whole
/// history would make the number fall more slowly every epoch and
/// would say nothing about the epoch that just ran.
pub fn BatchTrainer::epoch_loss(self : BatchTrainer) -> Double {
  let n = self.history.length()
  if n == 0 {
    return 0.0
  }
  let per_epoch = self.steps_per_epoch()
  let mut count = per_epoch
  if count > n {
    count = n
  }
  if count <= 0 {
    return 0.0
  }
  let mut acc = 0.0
  for i = 0; i < count; i = i + 1 {
    acc = acc + self.history[n - count + i]
  }
  acc / count.to_double()
}

///|
/// The first recorded loss.
pub fn BatchTrainer::initial_loss(self : BatchTrainer) -> Double {
  if self.history.length() == 0 {
    0.0
  } else {
    self.history[0]
  }
}

///|
/// The most recent loss.
pub fn BatchTrainer::final_loss(self : BatchTrainer) -> Double {
  if self.history.length() == 0 {
    0.0
  } else {
    self.history[self.history.length() - 1]
  }
}

///|
/// The loss at every step so far.
pub fn BatchTrainer::loss_history(self : BatchTrainer) -> Array[Double] {
  self.history.copy()
}

///|
/// Copy the host-side parameter tensors into the binding.
pub fn BatchTrainer::sync_params(self : BatchTrainer) -> Unit {
  for i = 0; i < self.params.length(); i = i + 1 {
    self.binding.set_param(i, self.params[i])
  }
}
