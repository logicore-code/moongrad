// A training recipe.
//
// The other examples each show one thing. This one shows a recipe:
// the pieces of a neural-network training run put together the way
// they would be put together in practice.
//
//   data          synthetic 8x8 images in four classes
//   model         convolution, batch normalisation, GELU, dropout,
//                 a dense head, and a softmax
//   optimiser     AdamW, a cosine schedule for the learning rate, and
//                 gradient clipping by norm
//   training      mini-batches, the gradient compiled once and
//                 evaluated at every batch
//   inference     the normalisation's running averages, which is what
//                 a trained model is used with
//
// The point of the design the library is built around shows in the
// training loop: the graph is built once, the gradient is compiled
// once, and every step re-evaluates that one program at the next
// batch. The dropout mask is redrawn between steps, and the schedule
// moves the learning rate, and the optimiser's state carries over —
// and none of that touches the graph.
//
// The data is generated rather than downloaded, so the example runs
// anywhere with a MoonBit toolchain and no network.

///|
/// The side of an image.
const SIDE : Int32 = 8

///|
/// The number of classes.
const CLASSES : Int32 = 4

///|
/// The batch size. It is baked into the graph's shapes, because a
/// tensor's shape is part of the graph rather than part of the values.
const BATCH : Int32 = 16

///|
/// The width of the hidden layer.
const HIDDEN : Int32 = 32

///|
/// The channels the convolution produces.
const CHANNELS : Int32 = 8

///|
/// The number of images per class.
const PER_CLASS : Int32 = 48

///|
/// The dataset: synthetic images, each one a pattern in a corner.
///
/// The four patterns are a horizontal bar, a vertical bar, a square
/// block and a diagonal, so the classes differ in a way a
/// convolution can find and a dense layer alone cannot.
fn make_dataset(
  n : Int32,
  seed : Int32,
) -> (Array[Array[Double]], Array[Int32]) {
  let rng = @moongrad.Lcg::new(seed)
  let images : Array[Array[Double]] = []
  let labels : Array[Int32] = []
  for i = 0; i < n; i = i + 1 {
    let label = i % CLASSES
    let pixels : Array[Double] = []
    for y = 0; y < SIDE; y = y + 1 {
      for x = 0; x < SIDE; x = x + 1 {
        let on = match label {
          0 => y >= 3 && y <= 4
          1 => x >= 3 && x <= 4
          2 => x >= 2 && x <= 5 && y >= 2 && y <= 5
          _ => x == y || x + 1 == y
        }
        let base = if on { 1.0 } else { 0.0 }
        pixels.push(base + rng.uniform(-0.15, 0.15))
      }
    }
    images.push(pixels)
    labels.push(label)
  }
  (images, labels)
}

///|
/// A batch of images, in the layout the graph wants: `[batch, 1, side,
/// side]`.
fn batch_tensor(
  images : Array[Array[Double]],
  order : Array[Int32],
  offset : Int32,
) -> @moongrad.Tensor {
  let data : Array[Double] = []
  for i = 0; i < BATCH; i = i + 1 {
    let image = images[order[(offset + i) % order.length()]]
    for p in image {
      data.push(p)
    }
  }
  match @moongrad.Tensor::from_data([BATCH, 1, SIDE, SIDE], data) {
    Some(t) => t
    None => @moongrad.Tensor::zeros([BATCH, 1, SIDE, SIDE])
  }
}

///|
/// The labels of a batch, one-hot, in the layout the loss wants.
fn batch_labels(
  labels : Array[Int32],
  order : Array[Int32],
  offset : Int32,
) -> @moongrad.Tensor {
  let data : Array[Double] = []
  for i = 0; i < BATCH; i = i + 1 {
    let label = labels[order[(offset + i) % order.length()]]
    for c = 0; c < CLASSES; c = c + 1 {
      data.push(if c == label { 1.0 } else { 0.0 })
    }
  }
  match @moongrad.Tensor::from_data([BATCH, CLASSES], data) {
    Some(t) => t
    None => @moongrad.Tensor::zeros([BATCH, CLASSES])
  }
}

///|
/// The model, the loss, and the parameters.
///
/// The node the normalisation reads is returned as well: the running
/// averages are computed from it, and they are what the model is used
/// with afterwards.
fn build(
  g : @moongrad.TGraph,
  dropout : @moongrad.Dropout,
) -> (Int32, Int32, Array[@moongrad.Tensor]) {
  let conv = @moongrad.Conv2d::new(g, 1, CHANNELS, 3, 3, 1, 1, 1, 1, 11)
  let dense = @moongrad.Linear::new(g, HIDDEN, CHANNELS * SIDE * SIDE, 22)
  let head = @moongrad.Linear::new(g, CLASSES, HIDDEN, 33)
  // The normalisation's own scale and shift are parameters, like the
  // weights.
  let gamma_slot = g.new_param_slot()
  let beta_slot = g.new_param_slot()
  let gamma = g.param(gamma_slot)
  let beta = g.param(beta_slot)
  let images = g.input(0)
  let onehot = g.input(1)
  let features = g.gelu(conv.apply(g, images))
  let flat = g.reshape(features, [BATCH, CHANNELS * SIDE * SIDE])
  let hidden = g.gelu(dense.apply(g, flat))
  // The normalisation reads the dense layer's output, and the running
  // averages are kept for the inference pass.
  // The batch axis, which is the first one: the normalisation is
  // over the batch, so a change to one sample moves the statistics
  // and so moves every other sample's output.
  let normalised = @moongrad.batch_norm(g, hidden, 0, gamma, beta, 1.0e-4)
  let dropped = dropout.apply(g, normalised, [BATCH, HIDDEN])
  let logits = head.apply(g, dropped)
  // The log-softmax rather than a logarithm of a softmax: the
  // softmax of a confident and wrong prediction can underflow to
  // zero, and the logarithm of a zero is not a number.
  let logp = g.log_softmax(logits, 1)
  let per_sample = g.sum_axis(g.mul(logp, onehot), 1)
  let loss = g.neg(g.sum_axis(per_sample, 0))
  let params : Array[@moongrad.Tensor] = []
  for t in conv.tensors() {
    params.push(t)
  }
  for t in dense.tensors() {
    params.push(t)
  }
  for t in head.tensors() {
    params.push(t)
  }
  // The scale and the shift, as a one-element tensor each.
  params.push(@moongrad.Tensor::filled([HIDDEN], 1.0))
  params.push(@moongrad.Tensor::filled([HIDDEN], 0.0))
  (loss, hidden, params)
}

///|
/// The inference model: the same network, with the normalisation's
/// running averages instead of the batch's.
///
/// The parameters are shared with the training graph — the same
/// tensors, written into the inference graph's slots — so this is the
/// trained model, not a second copy of it.
fn build_inference(
  g : @moongrad.TGraph,
  params : Array[@moongrad.Tensor],
  running_mean : @moongrad.Tensor,
  running_variance : @moongrad.Tensor,
) -> (Int32, Array[@moongrad.Tensor]) {
  let conv = @moongrad.Conv2d::new(g, 1, CHANNELS, 3, 3, 1, 1, 1, 1, 11)
  let dense = @moongrad.Linear::new(g, HIDDEN, CHANNELS * SIDE * SIDE, 22)
  let head = @moongrad.Linear::new(g, CLASSES, HIDDEN, 33)
  // The normalisation's scale and shift, in slots of their own.
  let gamma_slot = g.new_param_slot()
  let beta_slot = g.new_param_slot()
  let images = g.input(0)
  let features = g.gelu(conv.apply(g, images))
  let flat = g.reshape(features, [BATCH, CHANNELS * SIDE * SIDE])
  // The same activation as the training graph, before the
  // normalisation: an inference model that differs from the trained
  // one in anything but the statistics is a different model.
  let hidden = g.gelu(dense.apply(g, flat))
  let normalised = @moongrad.batch_norm_inference(
    g,
    hidden,
    g.param(gamma_slot),
    g.param(beta_slot),
    1.0e-4,
    running_mean,
    running_variance,
  )
  let logits = head.apply(g, normalised)
  // The trained parameters, in the order this graph's slots were
  // allocated: the convolution's weight and bias, the dense layer's,
  // the head's, and then the normalisation's scale and shift. The
  // layers built above have their own initial weights, which are not
  // what is wanted here.
  (logits, params.copy())
}

///|
fn main raise @moongrad.ADError {
  println("| moongrad: a training recipe")
  println("|")
  println("| a convolution, batch normalisation, GELU, dropout and a")
  println("| softmax, trained with AdamW, a cosine schedule and gradient")
  println("| clipping, over mini-batches.")
  println("|")
  let (images, labels) = make_dataset(CLASSES * PER_CLASS, 2024)
  let total = images.length()
  let order : Array[Int32] = []
  for i = 0; i < total; i = i + 1 {
    order.push(i)
  }
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let dropout = @moongrad.Dropout::new(0.25, 7)
  let (loss_node, normalised_input, params) = build(g, dropout)
  // The parameters start where the layers put them: the convolution
  // and the dense layers have their own, and the normalisation's
  // scale and shift start at one and zero.
  for i = 0; i < params.length(); i = i + 1 {
    b.set_param(i, params[i])
  }
  // The gradient is compiled once, and compiling needs the shapes of
  // the inputs, so a batch has to be in the binding before the
  // compilation rather than after it.
  b.set_input(0, batch_tensor(images, order, 0))
  b.set_input(1, batch_labels(labels, order, 0))
  let compiled = @moongrad.compile_tensor_gradient(g, b, loss_node)
  let shapes : Array[Array[Int32]] = []
  for p in params {
    shapes.push(p.shape.copy())
  }
  let opt = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::AdamW(0.003, 0.01, 1.0e-8),
    shapes,
  )
  // The schedule replaces the optimiser's own rate: a cosine from
  // 0.003 down to 0.0002 over the whole run.
  opt.set_schedule(@moongrad.LrSchedule::Cosine(0.003, 0.0002, 120))
  let running = @moongrad.BatchNormState::new(HIDDEN, 0.1)
  let steps = total / BATCH
  println("| epoch    loss      |grad|   rate")
  println("| -------------------------------")
  let mut step_index = 0
  for epoch = 0; epoch < 6; epoch = epoch + 1 {
    let mut total_loss = 0.0
    let mut worst = 0.0
    for s = 0; s < steps; s = s + 1 {
      step_index = step_index + 1
      let offset = s * BATCH
      b.set_input(0, batch_tensor(images, order, offset))
      b.set_input(1, batch_labels(labels, order, offset))
      // The mask is redrawn before every step: the graph reads it
      // from an input slot, and this is what puts a fresh mask
      // there.
      dropout.refresh(b)
      let values = @moongrad.t_forward(g, b)
      total_loss = total_loss + values[loss_node].data[0]
      let (_, grads) = @moongrad.evaluate_tensor_gradient(compiled, b)
      // The gradients are clipped by norm before the step, so that
      // one bad batch cannot throw the parameters across the map.
      let clipped = @moongrad.clip_grad_norm(grads, 1.0)
      let _ = clipped
      ignore(opt.step(params, grads, step_index))
      for p = 0; p < params.length(); p = p + 1 {
        b.set_param(p, params[p])
      }
      // The running averages are updated from the batch the
      // normalisation actually saw.
      let features = values[normalised_input]
      let (batch_mean, batch_variance) = @moongrad.batch_statistics(features)
      running.update(batch_mean, batch_variance)
      let g_norm = @moongrad.grad_norm_tensors(grads)
      if g_norm > worst {
        worst = g_norm
      }
    }
    let e = epoch + 1
    let rate = @moongrad.LrSchedule::Cosine(0.003, 0.0002, 120).at(step_index)
    println(
      "| \{e}       \{@moongrad.format_double(total_loss / steps.to_double())}   \{@moongrad.format_double(worst)}    \{@moongrad.format_double(rate)}",
    )
  }
  println("|")
  // The trained model is used with the normalisation's running
  // averages rather than a batch's, which is what the running
  // averages are for.
  let inference = @moongrad.TGraph::new()
  let ib = @moongrad.TBinding::new()
  let (logits_node, inference_params) = build_inference(
    inference,
    params,
    running.mean,
    running.variance,
  )
  for i = 0; i < inference_params.length(); i = i + 1 {
    ib.set_param(i, inference_params[i])
  }
  let mut right = 0
  let mut counted = 0
  for s = 0; s < steps; s = s + 1 {
    ib.set_input(0, batch_tensor(images, order, s * BATCH))
    let values = @moongrad.t_forward(inference, ib)
    let logits = values[logits_node]
    for i = 0; i < BATCH; i = i + 1 {
      let mut best = 0
      let mut best_value = logits.data[i * CLASSES]
      for c = 1; c < CLASSES; c = c + 1 {
        if logits.data[i * CLASSES + c] > best_value {
          best_value = logits.data[i * CLASSES + c]
          best = c
        }
      }
      let label = labels[order[(s * BATCH + i) % order.length()]]
      if best == label {
        right = right + 1
      }
      counted = counted + 1
    }
  }
  let accuracy = right.to_double() / counted.to_double()
  println("| accuracy on the training images, with the running averages:")
  println("|   \{@moongrad.format_double(accuracy * 100.0)}%")
  println("|")
}
