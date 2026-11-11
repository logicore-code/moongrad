// dataset.mbt
//
// Small dataset utilities.
//
// A dataset here is a table: rows of features, and one label per
// row. It is deliberately the plainest thing that can hold a
// training set, because everything in this file is a
// transformation of that table and nothing else — a shuffle, a
// slice, a batch, a fold.
//
// The package already has a `Dataset`, the one `train.mbt` uses to
// feed the training loop, which carries the number of inputs and
// outputs as well as the samples. The table here is a different
// thing and does the same job for a different caller, so it has a
// different name.
//
// The randomness is the generator in `nn.mbt`: a sixty-four bit
// linear congruential generator with a fixed seed, so a shuffle is
// the same on every machine. That matters more than it sounds. A
// training run that reshuffles differently on every machine is a
// training run nobody can reproduce, and a reproducibility bug in
// the data order is much harder to see than one in the arithmetic.

///|
/// A dense dataset: rows of features, and one label per row.
#derive(Debug)
pub(all) struct TabularDataset {
  /// The rows of features.
  x : Array[Array[Double]]
  /// The label of each row.
  y : Array[Double]
}

///|
pub extend TabularDataset with Debug::{to_repr}

///|
/// Raised when a dataset is not well formed.
#derive(Debug)
pub suberror DatasetError {
  /// The feature rows and the labels have different lengths.
  LengthMismatch(Int32, Int32)
  /// Two rows have different widths.
  RaggedRows(Int32, Int32)
}

///|
pub extend DatasetError with Debug::{to_repr}

///|
/// Human-readable form, in the style of `ADError::describe`.
pub fn DatasetError::describe(self : DatasetError) -> String {
  match self {
    LengthMismatch(rows, labels) => "\{rows} feature rows but \{labels} labels"
    RaggedRows(width, other) =>
      "the rows have different widths: \{width} and \{other}"
  }
}

///|
/// A dataset from parallel arrays.
///
/// Raises `DatasetError` when the two lengths disagree, and when the
/// rows have different widths. A dataset that has to be reshaped
/// before it can be used is a dataset that will be reshaped wrongly
/// somewhere, and the mistake shows up as a training run that does
/// not converge rather than as an error.
pub fn TabularDataset::new(
  x : Array[Array[Double]],
  y : Array[Double],
) -> TabularDataset raise DatasetError {
  if x.length() != y.length() {
    raise LengthMismatch(x.length(), y.length())
  }
  let width = if x.length() > 0 { x[0].length() } else { 0 }
  for row in x {
    if row.length() != width {
      raise RaggedRows(width, row.length())
    }
  }
  { x, y, }
}

///|
/// The number of rows.
pub fn TabularDataset::rows(self : TabularDataset) -> Int32 {
  self.x.length()
}

///|
/// The number of features in each row.
pub fn TabularDataset::features(self : TabularDataset) -> Int32 {
  if self.x.length() > 0 {
    self.x[0].length()
  } else {
    0
  }
}

///|
/// The same rows in a different order.
///
/// The order comes from a Fisher-Yates shuffle driven by the
/// generator in `nn.mbt`, so the same seed always gives the same
/// order. The rows themselves are not copied: the dataset that comes
/// back holds the same rows in a new order.
pub fn TabularDataset::shuffle(
  self : TabularDataset,
  seed : Int32,
) -> TabularDataset {
  let order = shuffled_indices(self.rows(), Lcg::new(seed))
  let x : Array[Array[Double]] = []
  let y : Array[Double] = []
  for i in order {
    x.push(self.x[i])
    y.push(self.y[i])
  }
  { x, y, }
}

///|
/// The first `n` rows.
///
/// A count larger than the dataset gives the whole dataset, and a
/// count of zero or less gives an empty one.
pub fn TabularDataset::take(self : TabularDataset, n : Int32) -> TabularDataset {
  let count = if n < 0 { 0 } else if n > self.rows() { self.rows() } else { n }
  let x : Array[Array[Double]] = []
  let y : Array[Double] = []
  for i = 0; i < count; i = i + 1 {
    x.push(self.x[i])
    y.push(self.y[i])
  }
  { x, y, }
}

///|
/// The rows `[index * size, index * size + size)`, as features and
/// labels.
///
/// The slice is clamped to the end of the dataset, so the last batch
/// of a dataset that does not divide evenly is shorter than the
/// others rather than running past the end.
pub fn TabularDataset::batch(
  self : TabularDataset,
  index : Int32,
  size : Int32,
) -> (Array[Array[Double]], Array[Double]) {
  let x : Array[Array[Double]] = []
  let y : Array[Double] = []
  if size <= 0 || index < 0 {
    return (x, y)
  }
  let start = index * size
  let mut i = start
  while i < start + size && i < self.rows() {
    x.push(self.x[i])
    y.push(self.y[i])
    i = i + 1
  }
  (x, y)
}

///|
/// The dataset split into `k` folds of roughly equal size.
///
/// The folds are contiguous and together hold every row exactly
/// once, so a cross-validation loop can train on all but one fold
/// and test on the one it left out. A dataset with fewer rows than
/// folds gives some empty folds rather than losing rows.
pub fn TabularDataset::kfold(
  self : TabularDataset,
  k : Int32,
) -> Array[TabularDataset] {
  let folds : Array[TabularDataset] = []
  if k <= 0 {
    return folds
  }
  let n = self.rows()
  for i = 0; i < k; i = i + 1 {
    let start = i * n / k
    let end = (i + 1) * n / k
    let x : Array[Array[Double]] = []
    let y : Array[Double] = []
    let mut j = start
    while j < end {
      x.push(self.x[j])
      y.push(self.y[j])
      j = j + 1
    }
    folds.push({ x, y, })
  }
  folds
}

///|
/// A synthetic regression dataset on `f(x) = 0.5x^2 - 1.5x + 2`.
///
/// The inputs are drawn uniformly from `[-3, 3]`, and every label
/// carries a little noise: a uniform perturbation of at most `0.05`.
/// That is small enough for the parabola to still be visible in a
/// scatter plot, and large enough that a fit does not collapse onto
/// the exact curve. The same seed gives the same dataset.
pub fn make_regression(n : Int32, seed : Int32) -> TabularDataset {
  let rng = Lcg::new(seed)
  let x : Array[Array[Double]] = []
  let y : Array[Double] = []
  for _i = 0; _i < n; _i = _i + 1 {
    let input = rng.uniform(-3.0, 3.0)
    let noise = rng.uniform(-0.05, 0.05)
    x.push([input])
    y.push(0.5 * input * input - 1.5 * input + 2.0 + noise)
  }
  { x, y, }
}

///|
/// A synthetic classification dataset of two clusters.
///
/// The clusters are two-dimensional: one sits around `(-2, -2)` and
/// the other around `(2, 2)`, with a spread of `0.3` in each
/// direction and the labels `0` and `1`. The centres are four apart
/// on each axis and the gap between the clusters is more than ten
/// spreads wide, so a straight line separates the two and a
/// two-layer network separates them without trouble.
pub fn make_clusters(n : Int32, seed : Int32) -> TabularDataset {
  let rng = Lcg::new(seed)
  let x : Array[Array[Double]] = []
  let y : Array[Double] = []
  let first = (n + 1) / 2
  for i = 0; i < n; i = i + 1 {
    let label = if i < first { 0.0 } else { 1.0 }
    let centre = if label == 0.0 { -2.0 } else { 2.0 }
    let a = centre + rng.uniform(-0.3, 0.3)
    let b = centre + rng.uniform(-0.3, 0.3)
    x.push([a, b])
    y.push(label)
  }
  { x, y, }
}
