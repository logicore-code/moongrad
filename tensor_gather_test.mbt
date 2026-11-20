// tensor_gather_test.mbt
//
// Gathering rows and putting them back.
//
// The two are inverses of each other, and the tests check both
// directions: the values come out where they should, and the
// gradients do too. A gather whose forward rule was right but whose
// backward rule was wrong would still train — towards the wrong
// thing — so the gradient is checked against a finite difference
// rather than against the forward rule alone.

///|
/// A tensor of the given shape and values.
fn tensor_of(shape : Array[Int32], values : Array[Double]) -> @moongrad.Tensor {
  match @moongrad.Tensor::from_data(shape, values) {
    Some(t) => t
    None => @moongrad.Tensor::zeros(shape)
  }
}

///|
/// A binding with one input of the given shape.
fn gather_binding(
  shape : Array[Int32],
  values : Array[Double],
) -> @moongrad.TBinding {
  let b = @moongrad.TBinding::new()
  b.set_input(0, tensor_of(shape, values))
  b
}

///|
test "gathering_takes_the_rows_the_index_names" {
  let x = tensor_of([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let index = @moongrad.Tensor::vector([2.0, 0.0])
  let out = match @moongrad.t_gather_rows(x, index) {
    Some(t) => t
    None => {
      check(false, "the gather should have worked")
      return
    }
  }
  assert_eq(out.shape.length(), 2)
  assert_eq(out.shape[0], 2)
  assert_eq(out.shape[1], 2)
  assert_close(out.data[0], 5.0, 1.0e-12, "the first element of row two")
  assert_close(out.data[1], 6.0, 1.0e-12, "the second element of row two")
  assert_close(out.data[2], 1.0, 1.0e-12, "the first element of row zero")
  assert_close(out.data[3], 2.0, 1.0e-12, "the second element of row zero")
}

///|
test "a_gather_keeps_the_rows_in_the_order_the_index_gives" {
  let x = tensor_of([3, 1], [10.0, 20.0, 30.0])
  let index = @moongrad.Tensor::vector([1.0, 1.0, 0.0])
  let out = match @moongrad.t_gather_rows(x, index) {
    Some(t) => t
    None => {
      check(false, "the gather should have worked")
      return
    }
  }
  assert_close(out.data[0], 20.0, 1.0e-12, "the first gathered row")
  assert_close(out.data[1], 20.0, 1.0e-12, "the repeated row")
  assert_close(out.data[2], 10.0, 1.0e-12, "the third gathered row")
}

///|
test "a_gather_refuses_an_index_out_of_range" {
  // A gather that quietly dropped rows would change the size of the
  // result, and the caller indexing into it would not know which rows
  // it had lost.
  let x = tensor_of([2, 1], [1.0, 2.0])
  let index = @moongrad.Tensor::vector([0.0, 5.0])
  let out = @moongrad.t_gather_rows(x, index)
  check(out is None, "an index out of range should be refused")
}

///|
test "a_gather_refuses_an_index_that_is_not_a_whole_number" {
  let x = tensor_of([2, 1], [1.0, 2.0])
  let index = @moongrad.Tensor::vector([0.5])
  let out = @moongrad.t_gather_rows(x, index)
  check(out is None, "a fractional index should be refused")
}

///|
test "a_scatter_puts_the_rows_back" {
  let x = tensor_of([2, 1], [5.0, 10.0])
  let index = @moongrad.Tensor::vector([2.0, 0.0])
  let out = match @moongrad.t_scatter_rows(x, index, [3, 1]) {
    Some(t) => t
    None => {
      check(false, "the scatter should have worked")
      return
    }
  }
  assert_eq(out.shape[0], 3)
  // The first row of x goes to the row the first index names, and the
  // second row of x goes to the row the second index names.
  assert_close(out.data[2], 5.0, 1.0e-12, "the first scattered row")
  assert_close(out.data[1], 0.0, 1.0e-12, "the row nothing was scattered to")
  assert_close(out.data[0], 10.0, 1.0e-12, "the second scattered row")
}

///|
test "a_scatter_sums_the_rows_that_share_a_destination" {
  let x = tensor_of([3, 1], [1.0, 2.0, 3.0])
  let index = @moongrad.Tensor::vector([0.0, 0.0, 1.0])
  let out = match @moongrad.t_scatter_rows(x, index, [2, 1]) {
    Some(t) => t
    None => {
      check(false, "the scatter should have worked")
      return
    }
  }
  assert_close(out.data[0], 3.0, 1.0e-12, "the summed row")
  assert_close(out.data[1], 3.0, 1.0e-12, "the row that was scattered once")
}

///|
test "a_gather_and_a_scatter_are_inverses" {
  let x = tensor_of([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let index = @moongrad.Tensor::vector([2.0, 0.0, 2.0])
  let gathered = match @moongrad.t_gather_rows(x, index) {
    Some(t) => t
    None => {
      check(false, "the gather should have worked")
      return
    }
  }
  let back = match @moongrad.t_scatter_rows(gathered, index, [3, 2]) {
    Some(t) => t
    None => {
      check(false, "the scatter should have worked")
      return
    }
  }
  // The row that was gathered twice is summed, so it is twice what it
  // was; the row that was gathered once is what it was; the row that
  // was not gathered at all is zero.
  assert_close(back.data[0], 1.0, 1.0e-12, "the row that was gathered once")
  assert_close(back.data[2], 0.0, 1.0e-12, "the row that was not gathered")
  assert_close(back.data[4], 10.0, 1.0e-12, "the row that was gathered twice")
  assert_close(back.data[5], 12.0, 1.0e-12, "the second element of that row")
}

///|
test "the_gradient_of_a_gather_is_a_scatter" {
  // f(x) = sum(gather(x, [1, 1]) * w). The gradient of x is the
  // weights put back where the rows came from, summed where the
  // indices agree.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let index = g.constant(@moongrad.Tensor::vector([1.0, 1.0]))
  let rows = g.gather_rows(x, index)
  let w = g.constant(@moongrad.Tensor::vector([2.0, 3.0]))
  let out = g.mul(rows, w)
  let loss = g.sum_axis(out, 0)
  let b = gather_binding([3], [1.0, 2.0, 3.0])
  let grads = @moongrad.t_grad(g, b, loss)
  // Both gathered rows are row one, so its gradient is the sum of the
  // two weights, and the other rows get nothing.
  assert_close(grads.input_grads[0].data[0], 0.0, 1.0e-12, "row zero")
  assert_close(grads.input_grads[0].data[1], 5.0, 1.0e-12, "row one")
  assert_close(grads.input_grads[0].data[2], 0.0, 1.0e-12, "row two")
}

///|
test "the_gradient_of_a_gather_matches_a_finite_difference" {
  // The gradient of a gather has to be right, not merely the forward
  // rule: a wrong backward rule would still train, towards the wrong
  // thing.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let index = g.constant(@moongrad.Tensor::vector([0.0, 2.0]))
  let rows = g.gather_rows(x, index)
  let squared = g.mul(rows, rows)
  let loss = g.sum_axis(squared, 0)
  let values = [1.5, -2.0, 0.5]
  let b = gather_binding([3], values)
  let grads = @moongrad.t_grad(g, b, loss)
  let h = 1.0e-6
  for i = 0; i < 3; i = i + 1 {
    let up = values.copy()
    let down = values.copy()
    up[i] = up[i] + h
    down[i] = down[i] - h
    let vu = @moongrad.t_forward(g, gather_binding([3], up))
    let vd = @moongrad.t_forward(g, gather_binding([3], down))
    let numeric = (vu[loss].data[0] - vd[loss].data[0]) / (2.0 * h)
    assert_close(
      grads.input_grads[0].data[i],
      numeric,
      1.0e-5,
      "the gather gradient",
    )
  }
}

///|
test "the_gradient_of_a_scatter_is_a_gather" {
  // f(x) = sum(scatter(x, [1, 0])). The gradient of x is the
  // incoming gradient read back along the index.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let index = g.constant(@moongrad.Tensor::vector([1.0, 0.0]))
  let scattered = g.scatter_rows(x, index, [2])
  let loss = g.sum_axis(scattered, 0)
  let b = gather_binding([2], [3.0, 4.0])
  let grads = @moongrad.t_grad(g, b, loss)
  assert_close(grads.input_grads[0].data[0], 1.0, 1.0e-12, "row zero")
  assert_close(grads.input_grads[0].data[1], 1.0, 1.0e-12, "row one")
}

///|
test "a_gather_can_read_a_different_row_at_a_different_binding" {
  // The index is a node, so the same program can gather different
  // rows at different bindings. That is what makes it usable for a
  // lookup table whose index comes from the data.
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let index = g.input(1)
  let rows = g.gather_rows(x, index)
  let b = @moongrad.TBinding::new()
  b.set_input(0, tensor_of([3, 1], [1.0, 2.0, 3.0]))
  b.set_input(1, @moongrad.Tensor::vector([0.0]))
  let first = @moongrad.t_forward(g, b)
  assert_close(first[rows].data[0], 1.0, 1.0e-12, "the first row")
  b.set_input(1, @moongrad.Tensor::vector([2.0]))
  let second = @moongrad.t_forward(g, b)
  assert_close(second[rows].data[0], 3.0, 1.0e-12, "the third row")
}
