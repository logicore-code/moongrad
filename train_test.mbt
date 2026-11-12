// train_test.mbt
//
// The training loop.
//
// The tests that matter are the ones that would fail if the loop had
// a wiring mistake rather than a numerical one. A loop that forgets
// to refresh the parameter binding trains forever on stale weights
// and produces a perfectly flat loss curve, which is easy to
// mistake for a learning-rate problem. `training_actually_moves_the
// _loss` exists to rule that out.

///|
/// A small well-separated dataset: two clusters on the unit square.
fn toy_dataset() -> @moongrad.Dataset {
  let xs : Array[Array[Double]] = [
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [0.1, 0.1],
    [0.9, 0.9],
    [1.0, 0.9],
    [0.9, 1.0],
    [1.0, 1.0],
  ]
  let ys : Array[Double] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
  let d = @moongrad.Dataset::new(xs, ys, 1)
  match d {
    Some(v) => v
    None => {
      check(false, "the toy dataset is well formed")
      @moongrad.Dataset::{ samples: [], n_in: 2, n_out: 1, }
    }
  }
}

///|
test "dataset_rejects_mismatched_lengths" {
  let bad = @moongrad.Dataset::new([[1.0], [2.0]], [0.0], 1)
  let rejected = match bad {
    None => true
    Some(_) => false
  }
  check(rejected, "two feature rows and one label is not a dataset")
}

///|
test "dataset_rejects_a_ragged_feature_matrix" {
  let bad = @moongrad.Dataset::new([[1.0, 2.0], [3.0]], [0.0, 1.0], 1)
  let rejected = match bad {
    None => true
    Some(_) => false
  }
  check(rejected, "a ragged feature matrix must be rejected, not padded")
}

///|
test "dataset_records_its_shape" {
  let d = toy_dataset()
  assert_eq(d.n_in, 2)
  assert_eq(d.n_out, 1)
  assert_eq(d.samples.length(), 8)
}

///|
test "shuffled_indices_are_a_permutation" {
  let idx = @moongrad.shuffled_indices(20, @moongrad.Lcg::new(3))
  assert_eq(idx.length(), 20)
  let mut seen = 0
  for i in idx {
    seen = seen + (1 << i)
  }
  // Every index set exactly once means the bitmask covers all 20.
  assert_eq(seen, (1 << 20) - 1)
}

///|
test "shuffled_indices_are_deterministic_for_a_seed" {
  let a = @moongrad.shuffled_indices(10, @moongrad.Lcg::new(7))
  let b = @moongrad.shuffled_indices(10, @moongrad.Lcg::new(7))
  for i = 0; i < 10; i = i + 1 {
    assert_eq(a[i], b[i])
  }
}

///|
test "shuffled_indices_actually_reorder" {
  // A shuffle that returns the identity every time is a plausible
  // bug in Fisher-Yates and would quietly bias any batched
  // training.
  let idx = @moongrad.shuffled_indices(50, @moongrad.Lcg::new(11))
  let mut moved = 0
  for i = 0; i < 50; i = i + 1 {
    if idx[i] != i {
      moved = moved + 1
    }
  }
  check(moved > 30, "a shuffle should move most elements, moved \{moved}")
}

///|
test "training_actually_moves_the_loss" {
  // The regression this catches: a loop that updates the host-side
  // parameters but never refreshes the binding, so the graph keeps
  // reading the initial weights. The loss stays exactly flat.
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 8, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    42,
  )
  let first = trainer.train_step(-1.0)
  for _i = 0; _i < 300; _i = _i + 1 {
    ignore(trainer.train_step(-1.0))
  }
  let last = trainer.train_step(-1.0)
  check(
    last.loss < first.loss * 0.2,
    "loss should fall by 5x: first \{first.loss}, last \{last.loss}",
  )
  check(last.ok, "the run should stay finite throughout")
}

///|
test "loss_history_has_one_entry_per_step" {
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 4, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    1,
  )
  for _i = 0; _i < 17; _i = _i + 1 {
    ignore(trainer.train_step(-1.0))
  }
  assert_eq(trainer.loss_history().length(), 17)
  assert_eq(trainer.step_count, 17)
}

///|
test "loss_history_is_non_increasing_on_average" {
  // Not a strict decrease: a single Adam step can overshoot. The
  // mean of the first fifth and the last fifth is the honest
  // measure of whether the optimiser is working at all.
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 6, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    9,
  )
  for _i = 0; _i < 200; _i = _i + 1 {
    ignore(trainer.train_step(-1.0))
  }
  let h = trainer.loss_history()
  let n = h.length()
  let head = mean_slice(h, 0, n / 5)
  let tail = mean_slice(h, n - n / 5, n)
  check(tail < head, "late loss \\{tail} should be below early loss \\{head}")
}

///|
fn mean_slice(xs : Array[Double], lo : Int32, hi : Int32) -> Double {
  let mut acc = 0.0
  for i = lo; i < hi; i = i + 1 {
    acc = acc + xs[i]
  }
  if hi <= lo {
    return 0.0
  }
  acc / (hi - lo).to_double()
}

///|
test "binary_cross_entropy_training_reduces_the_loss" {
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 6, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Sigmoid],
    d,
    @moongrad.LossKind::Bce,
    @moongrad.Optimizer::Adam(0.05),
    4,
  )
  let first = trainer.train_step(-1.0)
  for _i = 0; _i < 250; _i = _i + 1 {
    ignore(trainer.train_step(-1.0))
  }
  let last = trainer.train_step(-1.0)
  check(
    last.loss < first.loss,
    "BCE should fall: first \{first.loss}, last \{last.loss}",
  )
}

///|
test "gradient_norm_falls_as_the_run_converges" {
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 6, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    6,
  )
  let first = trainer.train_step(-1.0)
  for _i = 0; _i < 250; _i = _i + 1 {
    ignore(trainer.train_step(-1.0))
  }
  let last = trainer.train_step(-1.0)
  check(
    last.grad_norm < first.grad_norm,
    "gradient norm should fall: \{first.grad_norm} -> \{last.grad_norm}",
  )
}

///|
test "a_learning_rate_override_takes_effect" {
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 4, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    8,
  )
  // The first step uses the configured optimiser and moves the
  // weights. Every step after that uses the override, and an
  // override of zero means the weights cannot move at all, so the
  // loss from step 2 onwards must be constant.
  ignore(trainer.train_step(-1.0))
  let a = trainer.train_step(0.0).loss
  let b = trainer.train_step(0.0).loss
  let c = trainer.train_step(0.0).loss
  assert_close(b, a, 1.0e-12, "a zero learning rate freezes the loss (a vs b)")
  assert_close(c, b, 1.0e-12, "a zero learning rate freezes the loss (b vs c)")
}

///|
test "fit_runs_the_requested_number_of_steps" {
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 4, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    2,
  )
  let final_loss = trainer.fit(20)
  assert_eq(trainer.step_count, 20)
  check(@moongrad.is_finite(final_loss), "fit should return a finite loss")
}

///|
test "the_graph_is_built_once_and_reused" {
  // The whole point of compiling the gradient once: the graph must
  // not grow as training proceeds. If it did, a long run would
  // slowly run out of memory for no visible reason.
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 6, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    3,
  )
  let nodes_before = trainer.graph.len()
  let grad_before = trainer.grad_graph.len()
  for _i = 0; _i < 100; _i = _i + 1 {
    ignore(trainer.train_step(-1.0))
  }
  assert_eq(trainer.graph.len(), nodes_before)
  assert_eq(trainer.grad_graph.len(), grad_before)
}

///|
test "the_trainer_reports_its_initial_and_final_loss" {
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 4, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    5,
  )
  assert_eq(trainer.initial_loss(), 0.0)
  for _i = 0; _i < 10; _i = _i + 1 {
    ignore(trainer.train_step(-1.0))
  }
  check(
    trainer.initial_loss() != trainer.final_loss(),
    "the loss should have moved after ten steps",
  )
}

///|
test "initial_loss_of_an_untouched_trainer_is_zero" {
  let d = toy_dataset()
  let trainer = @moongrad.Trainer::new(
    [2, 4, 1],
    [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
    d,
    @moongrad.LossKind::Mse,
    @moongrad.Optimizer::Adam(0.05),
    5,
  )
  assert_eq(trainer.initial_loss(), 0.0)
  assert_eq(trainer.final_loss(), 0.0)
}

///|
test "the_tensor_trainer_handles_a_parameter_of_rank_two" {
  // The optimiser's state is sized by the number of *elements* in a
  // parameter, so a weight of shape (8, 4) needs thirty-two slots and
  // not twelve. Getting that wrong made the state run out in the
  // middle of a step.
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.TensorTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Adam(0.01),
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let x = graph.variable(0)
      let y = graph.variable(1)
      let layer = @moongrad.Linear::new(graph, 4, 8, 17)
      let pred = layer.apply(graph, x)
      let diff = graph.sub(pred, y)
      let squared = graph.mul(diff, diff)
      let total = graph.sum_axis(graph.sum_axis(squared, 1), 0)
      let loss = graph.div(
        total,
        graph.constant(@moongrad.Tensor::filled([], 6.0)),
      )
      ignore(y)
      (loss, layer.tensors())
    },
  )
  b.set_input(0, @moongrad.Tensor::zeros([6, 8]))
  b.set_input(1, @moongrad.Tensor::zeros([6, 4]))
  let report = trainer.step()
  check(report.ok, "the step should have succeeded")
}

///|
test "the_tensor_trainer_compiles_the_gradient_on_the_first_step" {
  // The backward rules bake the operand shapes into the program, so
  // the gradient cannot be compiled before the binding holds the
  // inputs. The trainer therefore compiles it on the first step, and
  // the order the caller sets the inputs up in stops mattering.
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.TensorTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.1),
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let x = graph.variable(0)
      let y = graph.variable(1)
      let layer = @moongrad.Linear::new(graph, 2, 2, 5)
      let pred = layer.apply(graph, x)
      let diff = graph.sub(pred, y)
      let squared = graph.mul(diff, diff)
      let loss = graph.div(
        graph.sum_axis(graph.sum_axis(squared, 1), 0),
        graph.constant(@moongrad.Tensor::filled([], 4.0)),
      )
      ignore(y)
      (loss, layer.tensors())
    },
  )
  // The inputs are set after the trainer is built.
  b.set_input(0, @moongrad.Tensor::filled([4, 2], 1.0))
  b.set_input(1, @moongrad.Tensor::filled([4, 2], 0.5))
  let report = trainer.step()
  check(report.ok, "the step should have succeeded")
  check(report.loss > 0.0, "the loss should be a number")
}

///|
test "the_tensor_trainer_reports_a_change_of_shape" {
  // A program compiled for one set of shapes is not valid for
  // another, and evaluating it anyway would quietly produce the
  // gradient of a different function — a zero gradient, in fact,
  // which looks like a network that is not learning.
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.TensorTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.1),
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let x = graph.variable(0)
      let y = graph.variable(1)
      let layer = @moongrad.Linear::new(graph, 2, 2, 5)
      let pred = layer.apply(graph, x)
      let diff = graph.sub(pred, y)
      let squared = graph.mul(diff, diff)
      let loss = graph.div(
        graph.sum_axis(graph.sum_axis(squared, 1), 0),
        graph.constant(@moongrad.Tensor::filled([], 4.0)),
      )
      ignore(y)
      (loss, layer.tensors())
    },
  )
  b.set_input(0, @moongrad.Tensor::filled([4, 2], 1.0))
  b.set_input(1, @moongrad.Tensor::filled([4, 2], 0.5))
  check(trainer.step().ok, "the first step should have succeeded")
  // Now the inputs change shape. The shapes are baked into the
  // compiled program, so the program is compiled again rather than
  // evaluated against shapes it was not built for — and the run
  // carries on.
  b.set_input(0, @moongrad.Tensor::filled([3, 2], 1.0))
  b.set_input(1, @moongrad.Tensor::filled([3, 2], 0.5))
  let report = trainer.step()
  check(report.ok, "a change of shape should recompile, not stop the run")
  check(report.loss > 0.0, "the loss should be a number")
}

///|
test "the_tensor_trainer_refuses_to_train_on_inputs_that_were_never_set" {
  // The graph declares inputs and the binding has none, so there is
  // no gradient to apply. Reporting that is better than moving the
  // parameters against nothing.
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.TensorTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.1),
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let x = graph.variable(0)
      let y = graph.variable(1)
      let layer = @moongrad.Linear::new(graph, 2, 2, 5)
      let pred = layer.apply(graph, x)
      let diff = graph.sub(pred, y)
      let squared = graph.mul(diff, diff)
      let loss = graph.div(
        graph.sum_axis(graph.sum_axis(squared, 1), 0),
        graph.constant(@moongrad.Tensor::filled([], 4.0)),
      )
      ignore(y)
      (loss, layer.tensors())
    },
  )
  let report = trainer.step()
  check(!report.ok, "training on no inputs should be refused")
}
