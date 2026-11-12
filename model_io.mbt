// model_io.mbt
//
// Model parameters as JSON.
//
// A model is a list of named tensors. The format is deliberately
// plain — the shape and the data of each parameter, in a fixed
// order — so that the same text reads back the same model:
//
//     {"params": [{"name": "w", "shape": [2, 3], "data": [1.0, 2.0, ...]}]}
//
// The numbers round-trip exactly. A double is written in the
// shortest form that reads back as the same double, and the reader
// keeps the written representation rather than re-deriving one, so a
// parameter that came out of a training run comes back bit for bit.
//
// There is no file here, and that is a deliberate stopping point.
// MoonBit's core library has no synchronous file API — the one in
// `moonbitlang/core/async/fs` is experimental and builds for some
// targets only — so this module stops at the string. Writing it to
// disk is the caller's business, and the string is the part worth
// getting right.

///|
/// A named parameter.
#derive(Debug)
pub(all) struct NamedParam {
  name : String
  value : Tensor
}

///|
pub extend NamedParam with Debug::{to_repr}

///|
/// A whole model: named parameters in a fixed order.
#derive(Debug)
pub(all) struct Model {
  params : Array[NamedParam]
}

///|
pub extend Model with Debug::{to_repr}

///|
/// What can go wrong reading a model back.
#derive(Debug)
pub suberror ModelError {
  /// The text was not valid JSON.
  BadJson(String)
  /// The JSON was valid but was not a model.
  BadModel(String)
  /// A parameter's data did not have the length its shape implies.
  ShapeMismatch(String)
  /// A dimension was negative.
  NegativeDimension(String)
}

///|
pub extend ModelError with Debug::{to_repr}

///|
/// A human-readable description of the failure.
pub fn ModelError::describe(self : ModelError) -> String {
  match self {
    BadJson(m) => "the text is not valid JSON: \{m}"
    BadModel(m) => "the JSON is not a model: \{m}"
    ShapeMismatch(m) => "a parameter's data does not match its shape: \{m}"
    NegativeDimension(m) => "a dimension is negative: \{m}"
  }
}

///|
/// Build a model from named parameters.
pub fn Model::new(params : Array[NamedParam]) -> Model {
  { params, }
}

///|
/// The model's parameters, in order.
pub fn Model::names(self : Model) -> Array[String] {
  let out : Array[String] = []
  for p in self.params {
    out.push(p.name)
  }
  out
}

///|
/// Look a parameter up by name.
pub fn Model::get(self : Model, name : String) -> Tensor? {
  for p in self.params {
    if p.name == name {
      return Some(p.value)
    }
  }
  None
}

///|
/// Replace a parameter's value.
///
/// Returns `false` when there is no parameter with that name, in
/// which case the model is unchanged.
pub fn Model::set(self : Model, name : String, value : Tensor) -> Bool {
  for i = 0; i < self.params.length(); i = i + 1 {
    if self.params[i].name == name {
      self.params[i] = { name, value, }
      return true
    }
  }
  false
}

///|
/// Write one double.
///
/// The three values JSON has no literal for — the infinities and the
/// not-a-number — are written as strings, and the sign of zero is
/// written out, because both of those are distinctions a model can
/// depend on and a round trip has to preserve.
fn write_double(x : Double) -> @json.Json {
  if x != x {
    return @json.Json::string("NaN")
  }
  if x == @double.infinity {
    return @json.Json::string("Infinity")
  }
  if x == @double.neg_infinity {
    return @json.Json::string("-Infinity")
  }
  if x == 0.0 && 1.0 / x < 0.0 {
    return @json.Json::number(x, repr="-0.0")
  }
  @json.Json::number(x)
}

///|
/// Read one double back, undoing `write_double`.
fn read_double(j : @json.Json) -> Double? {
  if j is String(s) {
    if s == "NaN" {
      return Some(@double.not_a_number)
    }
    if s == "Infinity" {
      return Some(@double.infinity)
    }
    if s == "-Infinity" {
      return Some(@double.neg_infinity)
    }
    return None
  }
  if j is Number(n, ..) {
    return Some(n)
  }
  None
}

///|
/// Serialise the model.
pub fn Model::to_json(self : Model) -> String {
  let entries : Array[@json.Json] = []
  for p in self.params {
    let shape : Array[@json.Json] = []
    for d in p.value.shape {
      shape.push(@json.Json::number(d.to_double()))
    }
    let data : Array[@json.Json] = []
    for x in p.value.data {
      data.push(write_double(x))
    }
    let fields = Map([
      ("name", @json.Json::string(p.name)),
      ("shape", @json.Json::array(shape)),
      ("data", @json.Json::array(data)),
    ])
    entries.push(@json.Json::object(fields))
  }
  let root = Map([("params", @json.Json::array(entries))])
  @json.Json::object(root).stringify()
}

///|
/// Read a model back.
///
/// The reader keeps the written representation of every number, so
/// the parameters come back exactly as they were written.
pub fn Model::from_json(text : String) -> Model raise ModelError {
  let tree = @json.parse(text, preserve_number_repr=true) catch {
    e => raise ModelError::BadJson(e.to_string())
  }
  let root = if tree is Object(o) {
    o
  } else {
    raise ModelError::BadModel("the top level is not an object")
  }
  let entries = match root.get("params") {
    Some(v) =>
      if v is Array(a) {
        a
      } else {
        raise ModelError::BadModel("params is not an array")
      }
    None => raise ModelError::BadModel("there is no params array")
  }
  let params : Array[NamedParam] = []
  for entry in entries {
    let fields = if entry is Object(o) {
      o
    } else {
      raise ModelError::BadModel("a parameter is not an object")
    }
    let name = match fields.get("name") {
      Some(v) =>
        if v is String(s) {
          s
        } else {
          raise ModelError::BadModel("a parameter's name is not a string")
        }
      None => raise ModelError::BadModel("a parameter has no name")
    }
    let shape_items = match fields.get("shape") {
      Some(v) =>
        if v is Array(a) {
          a
        } else {
          raise ModelError::BadModel("a parameter's shape is not an array")
        }
      None => raise ModelError::BadModel("a parameter has no shape")
    }
    let shape : Array[Int32] = []
    for d in shape_items {
      let extent = if d is Number(n, ..) {
        n
      } else {
        raise ModelError::BadModel("a shape entry is not a number")
      }
      if extent < 0.0 {
        raise ModelError::NegativeDimension(name)
      }
      // A dimension is a whole number of elements. A fractional one
      // is not a shape that could have been written, and truncating
      // it would accept a file that describes nothing.
      if extent != extent.floor() {
        raise ModelError::BadModel("\{name}: a dimension is not a whole number")
      }
      shape.push(extent.to_int())
    }
    let data_items = match fields.get("data") {
      Some(v) =>
        if v is Array(a) {
          a
        } else {
          raise ModelError::BadModel("a parameter's data is not an array")
        }
      None => raise ModelError::BadModel("a parameter has no data")
    }
    let data : Array[Double] = []
    for x in data_items {
      match read_double(x) {
        Some(n) => data.push(n)
        None => raise ModelError::BadModel("a data entry is not a number")
      }
    }
    // The element count is computed as a double, because the product
    // of a shape's dimensions can be larger than a machine integer
    // and would otherwise wrap round to something that looks like a
    // plausible count.
    let mut count = 1.0
    for d in shape {
      count = count * d.to_double()
    }
    if count > 2147483647.0 {
      raise ModelError::ShapeMismatch(
        "\{name}: the shape implies more elements than can be held",
      )
    }
    if count != data.length().to_double() {
      raise ModelError::ShapeMismatch(
        "\{name}: the shape implies \{count} elements but there are \{data.length()}",
      )
    }
    let value = Tensor::zeros(shape)
    for i = 0; i < data.length(); i = i + 1 {
      value.data[i] = data[i]
    }
    params.push({ name, value, })
  }
  { params, }
}
