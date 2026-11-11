// model_io_test.mbt
//
// Model parameters as JSON.
//
// The property that matters here is that a model comes back exactly
// as it went in: the same shapes, the same numbers, bit for bit. A
// serialisation that changes a weight in the last place changes the
// model, and the change is invisible until the training run no
// longer reproduces — which is the worst time to find it.
//
// The other tests are about refusal. Text that is not a model, or
// whose data does not match its shape, has to be turned away rather
// than read as something close enough.

///|
/// A matrix, or a failure the test should not survive.
fn matrix_of(rows : Array[Array[Double]]) -> @moongrad.Tensor {
  match @moongrad.Tensor::matrix(rows) {
    Some(t) => t
    None => {
      check(false, "the matrix should build")
      @moongrad.Tensor::zeros([0, 0])
    }
  }
}

///|
/// A model with a few awkward numbers in it.
fn sample_model() -> @moongrad.Model {
  let weights = matrix_of([
    [0.1, -2.5, 0.0],
    [1.0 / 3.0, 1.0e30, -0.0],
    [0.1 + 0.2, 2.0e-300, 1.0e300],
  ])
  let bias = @moongrad.Tensor::vector([0.25, -0.75])
  let scalar = @moongrad.Tensor::scalar(5.0e-324)
  let params = [
    @moongrad.NamedParam::{ name: "w", value: weights, },
    @moongrad.NamedParam::{ name: "b", value: bias, },
    @moongrad.NamedParam::{ name: "s", value: scalar, },
  ]
  @moongrad.Model::new(params)
}

///|
/// The value of a parameter, read back from a model.
fn value_of(model : @moongrad.Model, name : String) -> @moongrad.Tensor? {
  model.get(name)
}

///|
test "a_model_round_trips_through_json" {
  let model = sample_model()
  let text = model.to_json()
  let back = @moongrad.Model::from_json(text)
  assert_eq(back.params.length(), 3)
  assert_eq(back.params[0].name, "w")
  assert_eq(back.params[0].value.shape.length(), 2)
  assert_eq(back.params[0].value.shape[0], 3)
  assert_eq(back.params[0].value.shape[1], 3)
  assert_eq(back.params[1].value.shape.length(), 1)
  // A scalar has no axes at all.
  assert_eq(back.params[2].value.shape.length(), 0)
  // Every element, including the ones that are awkward to write.
  let w = match value_of(back, "w") {
    Some(t) => t
    None => {
      check(false, "the weights should be there")
      return
    }
  }
  let original = match value_of(model, "w") {
    Some(t) => t
    None => return
  }
  for i = 0; i < w.data.length(); i = i + 1 {
    check(
      w.data[i] == original.data[i],
      "element \{i} should survive the round trip, got \{w.data[i]}",
    )
  }
  // The sign of a zero is a bit of the number like any other.
  check(1.0 / w.data[5] < 0.0, "the negative zero should still be negative")
  let s = match value_of(back, "s") {
    Some(t) => t
    None => return
  }
  check(s.data[0] == 5.0e-324, "the smallest double should survive")
}

///|
test "the_json_of_a_model_is_deterministic" {
  let model = sample_model()
  let first = model.to_json()
  let second = model.to_json()
  assert_eq(first, second)
  // And it is the text it claims to be: one object per parameter,
  // with the name, the shape and the data.
  check(
    first.has_prefix("{\"params\":[{\"name\":\"w\",\"shape\":[3,3],\"data\":["),
    "the json should start with the first parameter, got \{first}",
  )
}

///|
test "a_name_with_an_escape_survives_the_round_trip" {
  let model = @moongrad.Model::new([
    @moongrad.NamedParam::{
      name: "w\"one\\",
      value: @moongrad.Tensor::vector([1.0]),
    },
  ])
  let back = @moongrad.Model::from_json(model.to_json())
  assert_eq(back.params[0].name, "w\"one\\")
}

///|
test "the_values_json_has_no_number_for_survive_the_round_trip" {
  let model = @moongrad.Model::new([
    @moongrad.NamedParam::{
      name: "odd",
      value: @moongrad.Tensor::vector([1.0 / 0.0, -1.0 / 0.0, 0.0 / 0.0]),
    },
  ])
  let back = @moongrad.Model::from_json(model.to_json())
  let values = match back.get("odd") {
    Some(t) => t
    None => {
      check(false, "the parameter should be there")
      return
    }
  }
  assert_eq(values.data[0], 1.0 / 0.0)
  assert_eq(values.data[1], -1.0 / 0.0)
  check(values.data[2] != values.data[2], "the NaN should still be a NaN")
}

///|
test "a_thousand_random_doubles_survive_the_round_trip" {
  let rng = @moongrad.Lcg::new(20240917)
  let mut ok = 0
  for _i = 0; _i < 1000; _i = _i + 1 {
    // A wide spread of magnitudes, so that the short decimal and the
    // long one are both exercised.
    let a = rng.uniform(-1.0, 1.0)
    let b = rng.uniform(-1.0, 1.0)
    let scale = rng.uniform(-300.0, 300.0)
    let x = (a + b) * @math.pow(10.0, scale)
    let model = @moongrad.Model::new([
      @moongrad.NamedParam::{ name: "x", value: @moongrad.Tensor::vector([x]), },
    ])
    let back = @moongrad.Model::from_json(model.to_json())
    let value = match back.get("x") {
      Some(t) => t
      None => {
        check(false, "the parameter should be there")
        return
      }
    }
    if value.data[0] == x {
      ok = ok + 1
    } else {
      check(false, "the double should survive the round trip")
      return
    }
  }
  assert_eq(ok, 1000)
}

///|
test "from_json_rejects_malformed_json" {
  let raised = try {
    let _ = @moongrad.Model::from_json("{\"params\":[{\"name\":\"w\"")
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "from_json_rejects_a_parameter_whose_data_does_not_match_its_shape" {
  // The shape says six elements; the data has two.
  let text = "{\"params\":[{\"name\":\"w\",\"shape\":[2,3],\"data\":[1,2]}]}"
  let raised = try {
    let _ = @moongrad.Model::from_json(text)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "from_json_rejects_a_negative_dimension" {
  let text = "{\"params\":[{\"name\":\"w\",\"shape\":[-1,2],\"data\":[1,2]}]}"
  let raised = try {
    let _ = @moongrad.Model::from_json(text)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "get_finds_a_parameter" {
  let model = sample_model()
  let bias = match value_of(model, "b") {
    Some(t) => t
    None => {
      check(false, "the bias should be there")
      return
    }
  }
  assert_eq(bias.data.length(), 2)
  assert_eq(bias.data[0], 0.25)
}

///|
test "get_returns_nothing_for_a_missing_parameter" {
  let model = sample_model()
  let missing = model.get("there_is_no_such_parameter")
  let found = match missing {
    None => true
    Some(_) => false
  }
  assert_true(found)
}

///|
test "set_replaces_the_value_of_a_parameter" {
  let model = sample_model()
  let replaced = model.set("b", @moongrad.Tensor::vector([1.0, 2.0]))
  assert_true(replaced)
  let bias = match value_of(model, "b") {
    Some(t) => t
    None => {
      check(false, "the bias should be there")
      return
    }
  }
  assert_eq(bias.data[0], 1.0)
  assert_eq(bias.data[1], 2.0)
  // The other parameters are untouched.
  let weights = match value_of(model, "w") {
    Some(t) => t
    None => return
  }
  assert_eq(weights.data[0], 0.1)
}

///|
test "set_refuses_a_missing_parameter" {
  let model = sample_model()
  let changed = model.set(
    "there_is_no_such_parameter",
    @moongrad.Tensor::vector([0.0]),
  )
  assert_false(changed)
  assert_eq(model.params.length(), 3)
}

///|
