// A convolutional network, trained.
//
// The other examples in this repository are small on purpose: each
// one shows a single idea and its output fits on a screen. This one
// is here to show that the pieces fit together at a size where the
// fitting matters.
//
// The task: a 12x12 grayscale image with a bright square in one of
// four corners and noise on every pixel. The network has to say
// which corner.
//
// The model is two convolutions, a dense layer and a softmax head,
// trained with AdamW over mini-batches. The graph is built once and
// the gradient is compiled once; every step writes the next batch
// into the binding and evaluates that same gradient program there.
//
// That is what compiling a gradient into a program buys: the program
// does not have to be thrown away between batches, and the same
// compiled gradient would answer a Hessian-vector product at any of
// them.
//
// The data is generated rather than downloaded, so the example runs
// anywhere with a MoonBit toolchain and no network.

///|
/// The size of one image, in pixels.
const SIDE : Int32 = 12

///|
/// The side of the bright square.
const SQUARE : Int32 = 4

///|
/// The number of corners, which is also the number of classes.
const CLASSES : Int32 = 4

///|
/// The batch size. It is baked into the graph's shapes, because a
/// tensor's shape is part of the graph rather than part of the
/// values.
const BATCH : Int32 = 32

///|
/// The dataset: a bright square in one of the four corners of a
/// 12x12 image, with noise on every pixel.
fn make_dataset(
  n : Int32,
  seed : Int32,
) -> (Array[Array[Double]], Array[Int32]) {
  let rng = @moongrad.Lcg::new(seed)
  let images : Array[Array[Double]] = []
  let labels : Array[Int32] = []
  for i = 0; i < n; i = i + 1 {
    // Two bits: which row the square is in, and which column.
    let row = if rng.next_f64() < 0.5 { 0 } else { 1 }
    let col = if rng.next_f64() < 0.5 { 0 } else { 1 }
    let label = row * 2 + col
    let pixels : Array[Double] = []
    for y = 0; y < SIDE; y = y + 1 {
      for x = 0; x < SIDE; x = x + 1 {
        let in_row = if row == 0 { y < SQUARE } else { y >= SIDE - SQUARE }
        let in_col = if col == 0 { x < SQUARE } else { x >= SIDE - SQUARE }
        let bright = if in_row && in_col { 1.0 } else { 0.0 }
        // The noise is what makes this a learning problem rather
        // than a lookup.
        pixels.push(bright + rng.uniform(-0.6, 0.6))
      }
    }
    images.push(pixels)
    labels.push(label)
  }
  (images, labels)
}

///|
/// The network, and the loss.
///
/// The loss is the cross-entropy of the softmax head against a
/// one-hot label. The one-hot label is an input rather than a
/// constant, so the graph does not have to be rebuilt when the batch
/// changes.
fn build(g : @moongrad.TGraph) -> (Int32, Int32, Array[@moongrad.Tensor]) {
  let conv1 = @moongrad.Conv2d::new(g, 1, 16, 3, 3, 1, 1, 1, 1, 11)
  let conv2 = @moongrad.Conv2d::new(g, 16, 32, 3, 3, 2, 2, 1, 1, 22)
  let dense = @moongrad.Linear::new(g, 32, 32 * 6 * 6, 33)
  let head = @moongrad.Linear::new(g, CLASSES, 32, 44)
  let images = g.input(0)
  let labels = g.input(1)
  // 12x12 -> 12x12 -> 6x6, with sixteen and then thirty-two
  // channels.
  let h1 = g.relu(conv1.apply(g, images))
  let h2 = g.relu(conv2.apply(g, h1))
  let flat = g.reshape(h2, [BATCH, 32 * 6 * 6])
  let h3 = g.relu(dense.apply(g, flat))
  let logits = head.apply(g, h3)
  // The log-softmax rather than a logarithm of a softmax: the
  // softmax of a confident and wrong prediction can underflow to
  // zero, and the logarithm of a zero is not a number.
  let logp = g.log_softmax(logits, 1)
  let per_sample = g.sum_axis(g.mul(logp, labels), 1)
  let loss = g.neg(g.sum_axis(per_sample, 0))
  // The parameters in slot order, which is the order the graph
  // allocated them in and the order the optimiser expects.
  let params : Array[@moongrad.Tensor] = []
  for t in conv1.tensors() {
    params.push(t)
  }
  for t in conv2.tensors() {
    params.push(t)
  }
  for t in dense.tensors() {
    params.push(t)
  }
  for t in head.tensors() {
    params.push(t)
  }
  (loss, logits, params)
}

///|
/// Write one batch into the binding: the images in slot 0 and the
/// one-hot labels in slot 1.
fn fill_batch(
  b : @moongrad.TBinding,
  images : Array[Array[Double]],
  labels : Array[Int32],
  order : Array[Int32],
  offset : Int32,
) -> Unit {
  let pixels : Array[Double] = []
  let onehot : Array[Double] = []
  for i = 0; i < BATCH; i = i + 1 {
    let index = order[(offset + i) % order.length()]
    for p in images[index] {
      pixels.push(p)
    }
    let label = labels[index]
    for c = 0; c < CLASSES; c = c + 1 {
      onehot.push(if c == label { 1.0 } else { 0.0 })
    }
  }
  let image_tensor = match
    @moongrad.Tensor::from_data([BATCH, 1, SIDE, SIDE], pixels) {
    Some(t) => t
    None => return
  }
  let label_tensor = match
    @moongrad.Tensor::from_data([BATCH, CLASSES], onehot) {
    Some(t) => t
    None => return
  }
  b.set_input(0, image_tensor)
  b.set_input(1, label_tensor)
}

///|
/// The fraction of the given batches the network gets right.
fn accuracy(
  g : @moongrad.TGraph,
  b : @moongrad.TBinding,
  logits_node : Int32,
  images : Array[Array[Double]],
  labels : Array[Int32],
) -> Double {
  let order : Array[Int32] = []
  for i = 0; i < images.length(); i = i + 1 {
    order.push(i)
  }
  let mut right = 0
  let mut total = 0
  let mut offset = 0
  while offset < images.length() {
    fill_batch(b, images, labels, order, offset)
    let values = @moongrad.t_forward(g, b)
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
      if best == labels[order[(offset + i) % order.length()]] {
        right = right + 1
      }
      total = total + 1
    }
    offset = offset + BATCH
  }
  if total == 0 {
    return 0.0
  }
  right.to_double() / total.to_double()
}

///|
fn main {
  println("| moongrad: a convolutional network, trained")
  println("|")
  println("| The task: a bright square in one of four corners of a")
  println("| 12x12 image, with noise on every pixel.")
  println("|")
  let (images, labels) = make_dataset(256, 2024)
  let (test_images, test_labels) = make_dataset(256, 99)
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let (loss_node, logits_node, params) = build(g)
  for i = 0; i < params.length(); i = i + 1 {
    b.set_param(i, params[i])
  }
  // A first batch, so that the gradient can be compiled: the
  // compiler evaluates the primal once to learn the shapes.
  let order : Array[Int32] = []
  for i = 0; i < images.length(); i = i + 1 {
    order.push(i)
  }
  fill_batch(b, images, labels, order, 0)
  let compiled = @moongrad.compile_tensor_gradient(g, b, loss_node)
  let shapes : Array[Array[Int32]] = []
  for p in params {
    shapes.push(p.shape.copy())
  }
  let opt = @moongrad.ExtraOptState::new(
    @moongrad.ExtraOptimizer::AdamW(0.002, 0.01, 1.0e-8),
    shapes,
  )
  let steps_per_epoch = images.length() / BATCH
  let mut parameter_count = 0
  for p in params {
    parameter_count = parameter_count + p.data.length()
  }
  println(
    "| the network has \{parameter_count} parameters in \{params.length()} tensors,",
  )
  println("| trained on \{images.length()} images in batches of \{BATCH}.")
  println("|")
  println("| epoch    loss")
  println("| -----------------")
  for e = 0; e < 4; e = e + 1 {
    let mut total = 0.0
    for i = 0; i < steps_per_epoch; i = i + 1 {
      let offset = i * BATCH
      fill_batch(b, images, labels, order, offset)
      let values = @moongrad.t_forward(g, b)
      total = total + values[loss_node].data[0]
      let (_, param_grads) = @moongrad.evaluate_tensor_gradient(compiled, b)
      ignore(opt.step(params, param_grads, e * steps_per_epoch + i + 1))
      // The optimiser changed the host-side tensors, so the binding
      // has to be told.
      for p = 0; p < params.length(); p = p + 1 {
        b.set_param(p, params[p])
      }
    }
    let epoch = e + 1
    println(
      "| \{@moongrad.format_double(epoch.to_double())}      \{@moongrad.format_double(total / steps_per_epoch.to_double())}",
    )
  }
  println("|")
  let acc = accuracy(g, b, logits_node, test_images, test_labels)
  println(
    "| accuracy on 256 unseen images: \{@moongrad.format_double(acc * 100.0)}%",
  )
  println("|")
}
