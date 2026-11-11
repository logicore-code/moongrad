// dataset_test.mbt
//
// Small dataset utilities.
//
// The properties that matter here are about the rows rather than
// about the numbers: a shuffle has to keep every row and change the
// order, a fold has to hold every row exactly once, and a batch has
// to be the slice it claims to be. A dataset that quietly loses or
// duplicates a row is a training run that is quietly learning on the
// wrong data, and nothing in the loss curve says so.

///|
/// A small dataset: ten rows, two features each.
fn toy_table() -> @moongrad.TabularDataset {
  let x : Array[Array[Double]] = [
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [0.1, 0.1],
    [0.2, 0.2],
    [0.3, 0.2],
    [0.2, 0.3],
    [0.3, 0.3],
    [0.4, 0.4],
    [0.5, 0.4],
  ]
  let y : Array[Double] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  let mut made : @moongrad.TabularDataset? = None
  try {
    made = Some(@moongrad.TabularDataset::new(x, y))
  } catch {
    _ => ()
  } noraise {
    _ => ()
  }
  match made {
    Some(d) => d
    None => {
      check(false, "the toy dataset is well formed")
      @moongrad.TabularDataset::{ x: [], y: [], }
    }
  }
}

///|
/// The rows of a dataset, as a string, so two datasets can be
/// compared.
fn rows_of(data : @moongrad.TabularDataset) -> String {
  let buf = StringBuilder()
  for i = 0; i < data.rows(); i = i + 1 {
    for feature in data.x[i] {
      buf.write_string(feature.to_string())
      buf.write_string(",")
    }
    buf.write_string(data.y[i].to_string())
    buf.write_string(" ")
  }
  buf.to_string()
}

///|
test "a_dataset_rejects_mismatched_lengths" {
  let rejected = try {
    let _ = @moongrad.TabularDataset::new([[1.0], [2.0]], [0.0])
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(rejected)
}

///|
test "a_dataset_rejects_ragged_rows" {
  let rejected = try {
    let _ = @moongrad.TabularDataset::new([[1.0, 2.0], [3.0]], [0.0, 1.0])
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(rejected)
}

///|
test "a_dataset_records_its_shape" {
  let data = toy_table()
  assert_eq(data.rows(), 10)
  assert_eq(data.features(), 2)
}

///|
test "shuffle_is_deterministic_for_a_seed" {
  let data = toy_table()
  let first = data.shuffle(7)
  let second = data.shuffle(7)
  assert_eq(rows_of(first), rows_of(second))
}

///|
test "shuffle_with_another_seed_gives_another_order" {
  let data = toy_table()
  let first = data.shuffle(7)
  let second = data.shuffle(8)
  let different = rows_of(first) != rows_of(second)
  assert_true(different)
}

///|
test "shuffle_keeps_every_row" {
  let data = toy_table()
  let shuffled = data.shuffle(3)
  assert_eq(shuffled.rows(), data.rows())
  // Every row of the original is in the shuffled dataset, and in it
  // only once.
  for j = 0; j < data.rows(); j = j + 1 {
    let mut hits = 0
    for i = 0; i < shuffled.rows(); i = i + 1 {
      if shuffled.x[i] == data.x[j] && shuffled.y[i] == data.y[j] {
        hits = hits + 1
      }
    }
    assert_eq(hits, 1)
  }
}

///|
test "take_returns_the_first_rows" {
  let data = toy_table()
  let taken = data.take(3)
  assert_eq(taken.rows(), 3)
  assert_eq(taken.features(), 2)
  assert_eq(taken.x[0], data.x[0])
  assert_eq(taken.x[2], data.x[2])
  // Taking more than there is gives the whole dataset.
  assert_eq(data.take(100).rows(), 10)
}

///|
test "batch_returns_the_right_slice" {
  let data = toy_table()
  let (x, y) = data.batch(1, 3)
  assert_eq(x.length(), 3)
  assert_eq(y.length(), 3)
  assert_eq(x[0], data.x[3])
  assert_eq(x[2], data.x[5])
  assert_eq(y[0], data.y[3])
}

///|
test "the_last_batch_is_shorter" {
  let data = toy_table()
  // Ten rows in batches of four: two full batches and a remainder.
  let (x, y) = data.batch(2, 4)
  assert_eq(x.length(), 2)
  assert_eq(y.length(), 2)
  assert_eq(x[0], data.x[8])
  assert_eq(x[1], data.x[9])
}

///|
test "kfold_holds_every_row_exactly_once" {
  let data = toy_table()
  let folds = data.kfold(3)
  assert_eq(folds.length(), 3)
  let mut total = 0
  for fold in folds {
    total = total + fold.rows()
  }
  assert_eq(total, 10)
  // Every row of the dataset is in one fold, and in one fold only.
  for i = 0; i < data.rows(); i = i + 1 {
    let mut hits = 0
    for fold in folds {
      for j = 0; j < fold.rows(); j = j + 1 {
        if fold.x[j] == data.x[i] && fold.y[j] == data.y[i] {
          hits = hits + 1
        }
      }
    }
    assert_eq(hits, 1)
  }
}

///|
test "make_regression_has_the_requested_number_of_rows" {
  let data = @moongrad.make_regression(40, 5)
  assert_eq(data.rows(), 40)
  assert_eq(data.features(), 1)
  assert_eq(data.y.length(), 40)
  // The labels follow the parabola, up to the noise.
  for i = 0; i < data.rows(); i = i + 1 {
    let input = data.x[i][0]
    let expected = 0.5 * input * input - 1.5 * input + 2.0
    check(
      (data.y[i] - expected).abs() <= 0.05,
      "the label should be the curve plus noise, got \{data.y[i]}",
    )
    check(input >= -3.0 && input <= 3.0, "the input should be in [-3, 3]")
  }
}

///|
test "make_clusters_is_two_separated_clusters" {
  let data = @moongrad.make_clusters(20, 11)
  assert_eq(data.rows(), 20)
  assert_eq(data.features(), 2)
  // Every label is zero or one, and the two clusters are far apart.
  for i = 0; i < data.rows(); i = i + 1 {
    check(
      data.y[i] == 0.0 || data.y[i] == 1.0,
      "the label should be zero or one",
    )
    let centre = if data.y[i] == 0.0 { -2.0 } else { 2.0 }
    check(
      (data.x[i][0] - centre).abs() <= 0.3,
      "the cluster should be around its centre",
    )
  }
}

///|
test "make_regression_is_the_same_for_a_seed" {
  let first = @moongrad.make_regression(10, 3)
  let second = @moongrad.make_regression(10, 3)
  assert_eq(rows_of(first), rows_of(second))
}
