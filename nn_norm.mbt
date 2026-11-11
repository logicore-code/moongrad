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
/// `[batch, features]` layout. In training mode the statistics come
/// from the batch itself, which is what couples the samples in the
/// gradient; at inference time they come from the running averages
/// the layer has been accumulating, and the samples are independent.
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
/// The mean of a tensor along one axis, keeping the axis.
pub fn mean_over(g : TGraph, x : Int32, axis : Int32) -> Int32 {
  g.mean_keep_axis(x, axis)
}
