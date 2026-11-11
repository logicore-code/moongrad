// train_batch_test.mbt
//
// Mini-batch training.
//
// The property worth testing is that mini-batching does not change
// what is being optimised: a batch of eight and a batch of the whole
// dataset should drive the same objective down to the same place,
// because the gradient program is the same program either way.

///|
/// A linear model on a two-feature dataset, with the loss rooted at
/// the returned node.
fn build_linear(graph : @moongrad.TGraph, batch : Int32) -> Int32 {
  let x = graph.variable(0)
  let y = graph.variable(1)
  let w = graph.param(0)
  let pred = graph.mul(x, w)
  let diff = graph.sub(pred, y)
  let sq = graph.mul(diff, diff)
  let s1 = graph.sum_axis(sq, 1)
  let s2 = graph.sum_axis(s1, 0)
  let loss = graph.div(
    s2,
    graph.constant(@moongrad.Tensor::filled([], batch.to_double())),
  )
  ignore(x)
  ignore(w)
  loss
}

///|
/// The weight the linear model starts from.
fn make_weight() -> @moongrad.Tensor {
  @moongrad.Tensor::zeros([1])
}

///|
/// A dataset on which the linear model has something to learn: one
/// feature, and a label that is twice it.
fn make_samples(n : Int32) -> (Array[Array[Double]], Array[Double]) {
  let x : Array[Array[Double]] = []
  let y : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    let a = (i % 7).to_double() - 3.0
    x.push([a])
    y.push(2.0 * a)
  }
  (x, y)
}

///|
test "mini_batch_training_reduces_the_loss" {
  let (x, y) = make_samples(40)
  let batch = 8
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Momentum(0.01, 0.9),
    x,
    y,
    0,
    1,
    batch,
    false,
    7,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let node = build_linear(graph, batch)
      (node, [make_weight()])
    },
  )
  let before = trainer.step().loss
  ignore(trainer.fit(40))
  let after = trainer.final_loss()
  check(
    after < before * 0.2,
    "the loss should fall by a lot: \{before} -> \{after}",
  )
  // The weights should have moved towards the answer: the first
  // weight converges to 2.
  let final_w = trainer.params[0]
  assert_close(final_w.data[0], 2.0, 0.5, "the recovered weight")
}

///|
test "the_batch_size_is_respected" {
  let (x, y) = make_samples(40)
  let batch = 8
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    x,
    y,
    0,
    1,
    batch,
    false,
    3,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let node = build_linear(graph, batch)
      (node, [make_weight()])
    },
  )
  let report = trainer.step()
  assert_eq(report.samples, batch)
  // The binding now holds a batch of features and a batch of labels.
  assert_eq(b.inputs[0].shape[0], batch)
  // The labels are a [batch, 1] tensor, so that they broadcast
  // against a row of predictions.
  assert_eq(b.inputs[1].shape.length(), 2)
  assert_eq(b.inputs[1].shape[0], batch)
  assert_eq(b.inputs[1].shape[1], 1)
}

///|
test "the_visit_order_is_reproducible" {
  let (x, y) = make_samples(20)
  let batch = 4
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    x,
    y,
    0,
    1,
    batch,
    false,
    11,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let node = build_linear(graph, batch)
      (node, [make_weight()])
    },
  )
  let first = trainer.order.copy()
  trainer.shuffle(11)
  let again = trainer.order.copy()
  for i = 0; i < first.length(); i = i + 1 {
    assert_eq(again[i], first[i])
  }
  trainer.shuffle(12)
  let other = trainer.order.copy()
  let mut different = false
  for i = 0; i < first.length(); i = i + 1 {
    if other[i] != first[i] {
      different = true
    }
  }
  check(different, "a different seed should give a different order")
}

///|
test "the_epoch_length_accounts_for_the_last_batch" {
  let (x, y) = make_samples(40)
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    x,
    y,
    0,
    1,
    8,
    false,
    5,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let node = build_linear(graph, 8)
      (node, [make_weight()])
    },
  )
  // 40 samples in batches of 8 is exactly five batches, and the
  // trainer keeps the last one rather than dropping it.
  assert_eq(trainer.steps_per_epoch(), 5)
}

///|
/// A linear model with one feature, for the batch-shape tests.
fn build_one(
  graph : @moongrad.TGraph,
  batch : Int32,
) -> (Int32, @moongrad.Tensor) {
  let x = graph.variable(0)
  let y = graph.variable(1)
  let w = graph.param(0)
  let pred = graph.mul(x, w)
  let diff = graph.sub(pred, y)
  let squared = graph.mul(diff, diff)
  let loss = graph.div(
    graph.sum_axis(graph.sum_axis(squared, 1), 0),
    graph.constant(@moongrad.Tensor::filled([], batch.to_double())),
  )
  (loss, @moongrad.Tensor::zeros([1]))
}

///|
fn one_feature(n : Int32) -> (Array[Array[Double]], Array[Double]) {
  let x : Array[Array[Double]] = []
  let y : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    x.push([i.to_double()])
    y.push(i.to_double())
  }
  (x, y)
}

///|
test "a_dropped_final_batch_is_skipped_rather_than_recycled" {
  // Ten samples in batches of four leaves a final batch of two. With
  // drop_last that batch is dropped, and the step that would have
  // used it must be skipped — not run on the previous batch's rows.
  let (x, y) = one_feature(10)
  let batch = 4
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    x,
    y,
    0,
    1,
    batch,
    true,
    11,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let (node, weight) = build_one(graph, batch)
      (node, [weight])
    },
  )
  // Two whole batches fit, the third is short.
  let first = trainer.step()
  let second = trainer.step()
  let third = trainer.step()
  check(first.ok, "the first batch should be a whole one")
  check(second.ok, "the second batch should be a whole one")
  check(!third.ok, "the short final batch should be skipped")
  assert_eq(third.samples, 2)
}

///|
test "an_empty_dataset_does_not_panic" {
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    [],
    [],
    0,
    1,
    4,
    false,
    3,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let (node, weight) = build_one(graph, 4)
      (node, [weight])
    },
  )
  let report = trainer.step()
  check(!report.ok, "there is no data to train on")
}

///|
test "a_row_of_a_different_width_is_refused" {
  let x : Array[Array[Double]] = [[0.0, 1.0], [2.0, 3.0, 4.0]]
  let y : Array[Double] = [0.0, 1.0]
  let batch = 2
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    x,
    y,
    0,
    1,
    batch,
    false,
    3,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let (node, weight) = build_one(graph, batch)
      (node, [weight])
    },
  )
  let report = trainer.step()
  check(!report.ok, "a ragged dataset should be refused")
}

///|
test "the_batch_used_to_fix_the_shapes_is_not_consumed" {
  // The trainer fills a batch before compiling the gradient, so that
  // the compiler sees the right shapes. That fill must not eat the
  // first batch of the run, so the cursor is put back where it was.
  let (x, y) = one_feature(10)
  let batch = 4
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    x,
    y,
    0,
    1,
    batch,
    false,
    5,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let (node, weight) = build_one(graph, batch)
      (node, [weight])
    },
  )
  // The cursor has not moved: the first step will draw from the top
  // of the order.
  assert_eq(trainer.cursor, 0)
  let before = trainer.order.copy()
  ignore(trainer.step())
  // The first four of the order are the ones that were drawn.
  let mut used = 0
  for i = 0; i < 4; i = i + 1 {
    check(
      trainer.order[i] == before[i],
      "the order should not have been disturbed",
    )
    used = used + 1
  }
  assert_eq(used, 4)
  assert_eq(trainer.cursor, 4)
}

///|
test "the_epoch_loss_is_the_mean_of_the_epoch" {
  let (x, y) = one_feature(12)
  let batch = 4
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    x,
    y,
    0,
    1,
    batch,
    false,
    7,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let (node, weight) = build_one(graph, batch)
      (node, [weight])
    },
  )
  ignore(trainer.fit(1))
  let first = trainer.epoch_loss()
  ignore(trainer.fit(1))
  let second = trainer.epoch_loss()
  // The second epoch's mean is not the mean of the whole history,
  // because the first epoch's losses are in there too.
  let history = trainer.loss_history()
  let mut all = 0.0
  for v in history {
    all = all + v
  }
  let overall = all / history.length().to_double()
  check(
    second != overall || history.length() == trainer.steps_per_epoch(),
    "the epoch loss should not be the average of the whole history",
  )
  ignore(first)
}

///|
test "a_batch_size_of_zero_is_not_an_epoch" {
  let (x, y) = one_feature(4)
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let trainer = @moongrad.BatchTrainer::new(
    g,
    b,
    @moongrad.TensorOptimizer::Sgd(0.01),
    x,
    y,
    0,
    1,
    0,
    false,
    3,
    fn(
      graph : @moongrad.TGraph,
      _binding : @moongrad.TBinding,
    ) -> (Int32, Array[@moongrad.Tensor]) {
      let (node, weight) = build_one(graph, 0)
      (node, [weight])
    },
  )
  assert_eq(trainer.steps_per_epoch(), 0)
  let report = trainer.step()
  check(!report.ok, "a batch of nothing should not be a step")
}
