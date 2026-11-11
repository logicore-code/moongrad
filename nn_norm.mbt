// nn_norm.mbt
//
// Normalisation.
//
// Both normalisation layers in common use are the same computation
// over a different axis. Layer normalisation centres each sample
// over its own features; batch normalisation centres each feature
// over the batch. Writing them once means the two cannot drift
// apart, and it means the gradient — which the compiler derives, so
// nobody writes it — is the same derivation in both cases.
//
// The computation is
//
//     out = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
//
// with the mean and the variance taken over one axis. The mean is
// kept with its axis, at length one, so it stays broadcastable
// against the tensor it came from; that is what lets the layer be
// built without knowing the tensor's shape.
//
// The gradient of this is not simple, and the reason it is correct
// anyway is the point of the whole library: the compiler
// differentiates the expression above by differentiating it, and the
// coupling between the batch and the mean falls out of that rather
// than being written down.

///|
/// Normalise a tensor along one axis.
///
/// `axis` is the axis the mean and variance are taken over. `gamma`
/// and `beta` are per-feature scale and shift; either may be omitted
/// by passing a constant one and a constant zero.
pub fn normalize(
  g : TGraph,
  x : Int32,
  axis : Int32,
  gamma : Int32,
  beta : Int32,
  eps : Double,
) -> Int32 {
  let mean = g.mean_keep_axis(x, axis)
  let centred = g.sub(x, mean)
  let squared = g.mul(centred, centred)
  let variance = g.mean_keep_axis(squared, axis)
  let shifted = g.add(variance, g.constant(Tensor::filled([], eps)))
  let inverse = g.div(g.constant(Tensor::filled([], 1.0)), g.sqrt(shifted))
  let scaled = g.mul(centred, inverse)
  let out = g.mul(scaled, gamma)
  g.add(out, beta)
}

///|
/// Layer normalisation: normalise each sample over its features.
///
/// `axis` is the feature axis, which is the last one for the usual
/// `[batch, features]` layout.
pub fn layer_norm(
  g : TGraph,
  x : Int32,
  axis : Int32,
  gamma : Int32,
  beta : Int32,
  eps : Double,
) -> Int32 {
  normalize(g, x, axis, gamma, beta, eps)
}

///|
/// Batch normalisation: normalise each feature over the batch.
///
/// `axis` is the batch axis, which is the first one for the usual
/// `[batch, features]` layout. The statistics come from the batch
/// itself, which is what couples the samples in the gradient: a
/// change to one sample moves the mean and the variance, and so
/// moves every other sample's output.
///
/// At inference time a batch is a single sample, and there is no
/// batch to take statistics over. `batch_norm_inference` takes them
/// from a `BatchNormState` instead, which is where the running
/// averages live.
pub fn batch_norm(
  g : TGraph,
  x : Int32,
  axis : Int32,
  gamma : Int32,
  beta : Int32,
  eps : Double,
) -> Int32 {
  normalize(g, x, axis, gamma, beta, eps)
}

///|
/// Batch normalisation with the statistics given rather than taken
/// from the batch.
///
/// `mean` and `variance` hold one entry per feature and are
/// broadcast against the `[batch, features]` activation, so the
/// samples are independent: no gradient crosses from one to another.
pub fn batch_norm_inference(
  g : TGraph,
  x : Int32,
  gamma : Int32,
  beta : Int32,
  eps : Double,
  mean : Tensor,
  variance : Tensor,
) -> Int32 {
  let centred = g.sub(x, g.constant(mean))
  let shifted = g.add(g.constant(variance), g.constant(Tensor::filled([], eps)))
  let inverse = g.div(g.constant(Tensor::filled([], 1.0)), g.sqrt(shifted))
  let scaled = g.mul(centred, inverse)
  let out = g.mul(scaled, gamma)
  g.add(out, beta)
}

///|
/// The running statistics of a batch-normalised layer.
///
/// The training form of the layer takes its statistics from the
/// batch it is given; at inference time it has to take them from
/// somewhere, and this is where they come from: an exponential
/// average of every batch seen so far.
#derive(Debug)
pub(all) struct BatchNormState {
  /// The running mean of each feature.
  mean : Tensor
  /// The running variance of each feature.
  variance : Tensor
  /// How much of each batch's statistics to keep.
  momentum : Double
  /// How many batches have been seen.
  mut batches : Int32
}

///|
pub extend BatchNormState with Debug::{to_repr}

///|
/// Fresh statistics for `features` features.
///
/// The variance starts at one and the mean at zero, which is the
/// usual choice: before any batch has been seen the layer should be
/// close to the identity.
pub fn BatchNormState::new(
  features : Int32,
  momentum : Double,
) -> BatchNormState {
  {
    mean: Tensor::zeros([features]),
    variance: Tensor::filled([features], 1.0),
    momentum,
    batches: 0,
  }
}

///|
/// Fold one batch's statistics into the running averages.
///
/// The batch's mean and variance are one entry per feature. The
/// update is the usual exponential average: the running value keeps
/// a fraction of itself and takes the rest from the batch. The first
/// batch is taken whole, because a running average that starts from
/// a guess takes a long time to forget the guess.
pub fn BatchNormState::update(
  self : BatchNormState,
  batch_mean : Array[Double],
  batch_variance : Array[Double],
) -> Unit {
  let n = batch_mean.length()
  if n != self.mean.data.length() {
    return
  }
  if self.batches == 0 {
    for i = 0; i < n; i = i + 1 {
      self.mean.data[i] = batch_mean[i]
      self.variance.data[i] = batch_variance[i]
    }
  } else {
    let keep = 1.0 - self.momentum
    for i = 0; i < n; i = i + 1 {
      self.mean.data[i] = keep * self.mean.data[i] +
        self.momentum * batch_mean[i]
      self.variance.data[i] = keep * self.variance.data[i] +
        self.momentum * batch_variance[i]
    }
  }
  self.batches = self.batches + 1
}

///|
/// The running mean and variance, as the pair the inference form
/// takes.
pub fn BatchNormState::statistics(self : BatchNormState) -> (Tensor, Tensor) {
  (self.mean, self.variance)
}

///|
/// The mean and variance of each feature of a batch.
///
/// The batch is `[batch, features]` and the statistics are taken over
/// the rows. This is the host-side counterpart of the graph's
/// `batch_norm`: the graph computes the same numbers inside itself
/// (which is what makes the gradient correct), and this computes
/// them outside it, where the running averages can be updated.
pub fn batch_statistics(x : Tensor) -> (Array[Double], Array[Double]) {
  let mean : Array[Double] = []
  let variance : Array[Double] = []
  if x.shape.length() != 2 {
    return (mean, variance)
  }
  let rows = x.shape[0]
  let features = x.shape[1]
  for f = 0; f < features; f = f + 1 {
    let mut total = 0.0
    for r = 0; r < rows; r = r + 1 {
      total = total + x.data[r * features + f]
    }
    let m = if rows > 0 { total / rows.to_double() } else { 0.0 }
    let mut sum = 0.0
    for r = 0; r < rows; r = r + 1 {
      let d = x.data[r * features + f] - m
      sum = sum + d * d
    }
    let v = if rows > 0 { sum / rows.to_double() } else { 0.0 }
    mean.push(m)
    variance.push(v)
  }
  (mean, variance)
}

///|
/// The mean of a tensor along one axis, keeping the axis.
pub fn mean_over(g : TGraph, x : Int32, axis : Int32) -> Int32 {
  g.mean_keep_axis(x, axis)
}
