// tensor_test.mbt
//
// The host-side matrix type. Small enough that a wrong answer is
// obvious, which is the right property for the type everything else
// indexes into.

///|
test "zeros_has_the_requested_shape" {
  let m = @moongrad.Matrix::zeros(3, 4)
  assert_eq(m.rows, 3)
  assert_eq(m.cols, 4)
  assert_eq(m.len(), 12)
  assert_eq(m.get(2, 3), 0.0)
}

///|
test "set_and_get_round_trip" {
  let m = @moongrad.Matrix::zeros(2, 3)
  m.set(1, 2, 7.5)
  assert_eq(m.get(1, 2), 7.5)
  assert_eq(m.get(1, 1), 0.0)
}

///|
test "indexing_is_row_major" {
  // (1,2) of a 2x3 is element 5 of the flat array. Getting this
  // wrong would silently transpose every matrix product in the
  // layer code, so it is worth pinning.
  let m = @moongrad.Matrix::zeros(2, 3)
  m.set(1, 2, 9.0)
  assert_eq(m.data[5], 9.0)
}

///|
test "identity_is_the_identity" {
  let m = @moongrad.Matrix::identity(3)
  for i = 0; i < 3; i = i + 1 {
    assert_eq(m.get(i, i), 1.0)
    for j = 0; j < 3; j = j + 1 {
      if i != j {
        assert_eq(m.get(i, j), 0.0)
      }
    }
  }
}

///|
test "from_array_rejects_a_length_mismatch" {
  let bad = @moongrad.Matrix::from_array(2, 2, [1.0, 2.0])
  let rejected = match bad {
    None => true
    Some(_) => false
  }
  check(rejected, "a 2x2 matrix needs four entries, not two")
}

///|
test "from_array_accepts_a_matching_length" {
  let ok = @moongrad.Matrix::from_array(2, 2, [1.0, 2.0, 3.0, 4.0])
  match ok {
    Some(m) => {
      assert_eq(m.get(0, 0), 1.0)
      assert_eq(m.get(1, 1), 4.0)
    }
    None => check(false, "from_array rejected a well-formed matrix")
  }
}

///|
test "matrix_product_matches_the_hand_computation" {
  // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
  let a = @moongrad.Matrix::from_flat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let b = @moongrad.Matrix::from_flat(2, 2, [5.0, 6.0, 7.0, 8.0])
  match a.mul(b) {
    Some(c) => {
      assert_eq(c.get(0, 0), 19.0)
      assert_eq(c.get(0, 1), 22.0)
      assert_eq(c.get(1, 0), 43.0)
      assert_eq(c.get(1, 1), 50.0)
    }
    None => check(false, "2x2 * 2x2 must multiply")
  }
}

///|
test "matrix_product_rejects_mismatched_shapes" {
  let a = @moongrad.Matrix::zeros(2, 3)
  let b = @moongrad.Matrix::zeros(4, 5)
  let r = a.mul(b)
  let rejected = match r {
    None => true
    Some(_) => false
  }
  check(rejected, "2x3 * 4x5 has no valid product")
}

///|
test "mul_t_is_the_transposed_product" {
  // [[1,2],[3,4]] * [[5,7],[6,8]]^T = [[19,22],[43,50]]
  let a = @moongrad.Matrix::from_flat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let b = @moongrad.Matrix::from_flat(2, 2, [5.0, 7.0, 6.0, 8.0])
  match a.mul_t(b) {
    Some(c) => {
      assert_eq(c.get(0, 0), 19.0)
      assert_eq(c.get(1, 1), 50.0)
    }
    None => check(false, "2x2 * 2x2^T must multiply")
  }
}

///|
test "transpose_round_trips" {
  let a = @moongrad.Matrix::from_flat(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let t = a.transpose()
  assert_eq(t.rows, 3)
  assert_eq(t.cols, 2)
  assert_eq(t.get(0, 1), 4.0)
  let back = t.transpose()
  for i = 0; i < a.len(); i = i + 1 {
    assert_eq(back.data[i], a.data[i])
  }
}

///|
test "element_wise_operations" {
  let a = @moongrad.Matrix::from_flat(2, 2, [1.0, 2.0, 3.0, 4.0])
  let b = @moongrad.Matrix::from_flat(2, 2, [10.0, 20.0, 30.0, 40.0])
  assert_eq(a.add(b).get(0, 0), 11.0)
  assert_eq(b.sub(a).get(1, 1), 36.0)
  assert_eq(a.scale(3.0).get(0, 1), 6.0)
  assert_eq(a.hadamard(a).get(1, 0), 9.0)
}

///|
test "sum_mean_and_norm" {
  let m = @moongrad.Matrix::from_flat(2, 2, [3.0, 4.0, 0.0, 0.0])
  assert_eq(m.sum(), 7.0)
  assert_eq(m.mean(), 1.75)
  assert_eq(m.norm(), 5.0)
}

///|
test "max_abs_finds_the_largest_magnitude" {
  let m = @moongrad.Matrix::from_flat(1, 3, [1.0, -9.0, 4.0])
  assert_eq(m.max_abs(), 9.0)
}

///|
test "all_finite_rejects_a_poisoned_matrix" {
  let good = @moongrad.Matrix::from_flat(1, 2, [1.0, 2.0])
  check(good.all_finite(), "1 and 2 are finite")
  let bad = @moongrad.Matrix::from_flat(1, 2, [1.0, 0.0 / 0.0])
  check(!bad.all_finite(), "NaN must be reported")
}

///|
test "column_builds_a_vector" {
  let v = @moongrad.Matrix::column([1.0, 2.0, 3.0])
  assert_eq(v.rows, 3)
  assert_eq(v.cols, 1)
  assert_eq(v.get(2, 0), 3.0)
}

///|
test "flat_round_trip_preserves_everything" {
  let a = @moongrad.Matrix::from_flat(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = @moongrad.Matrix::from_flat(3, 2, a.to_array())
  for i = 0; i < a.len(); i = i + 1 {
    assert_eq(b.data[i], a.data[i])
  }
}

///|
test "preview_truncates_wide_rows" {
  let m = @moongrad.Matrix::zeros(2, 10)
  let s = m.preview(3)
  check(s.contains("..."), "a truncated preview should say so: \{s}")
}

///|
test "format_double_rounds_to_four_places" {
  assert_eq(@moongrad.format_double(1.0 / 3.0), "0.3333")
  assert_eq(@moongrad.format_double(2.0), "2")
}
