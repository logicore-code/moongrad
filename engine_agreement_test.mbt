// engine_agreement_test.mbt
//
// The two engines, compared against each other.
//
// The scalar engine and the tensor engine are separate code. The
// scalar backward sweep emits expressions in a scalar graph and the
// tensor one emits expressions in a tensor graph, so each has its own
// set of derivative rules; the numerical table in `simplify.mbt` is a
// third thing, and it is what the forward-mode engine checks against
// rather than what the backward sweeps share.
//
// The claim that the two engines agree is therefore a claim about
// behaviour, not about shared code, and it is worth testing. The tests
// below take every operator the two engines have in common and
// compare the gradient they compute at the same point. A rule fixed
// in one engine and not the other shows up here.
//
// The tensor engine has operators the scalar one does not --
// convolution, normalisation, the broadcasting reductions -- and those
// are checked by the finite-difference tests in their own files.

///|
fn scalar_binding_at(v : Double) -> @moongrad.Binding {
  let b = @moongrad.Binding::new()
  b.set_var(0, v)
  b
}

///|
fn tensor_binding_at(v : Double) -> @moongrad.TBinding {
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([v]))
  b
}

///|
/// The gradient the scalar engine computes at `v`.
fn scalar_grad(
  build : (@moongrad.Graph, Int32) -> Int32,
  v : Double,
) -> Double raise @moongrad.ADError {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let y = build(g, x)
  let (grads, _params) = @moongrad.grad(g, scalar_binding_at(v), y)
  grads[0]
}

///|
/// The gradient the tensor engine computes at `v`.
fn tensor_grad(
  build : (@moongrad.TGraph, Int32) -> Int32,
  v : Double,
) -> Double {
  let g = @moongrad.TGraph::new()
  let x = g.input(0)
  let y = build(g, x)
  let grads = @moongrad.t_grad(g, tensor_binding_at(v), y)
  grads.input_grads[0].data[0]
}

///|
test "the_two_engines_agree_on_the_unary_operators" {
  // The point the two are compared at is one where the operators
  // differ from each other in shape: the derivative of a logarithm
  // and of a tangent are large there, and a rule that is wrong by a
  // factor would not hide.
  let point = 0.7
  let scalar_cases : Array[(String, (@moongrad.Graph, Int32) -> Int32)] = [
    ("neg", fn(g, x) { g.neg(x) }),
    ("exp", fn(g, x) { g.exp(x) }),
    ("ln", fn(g, x) { g.ln(x) }),
    ("sqrt", fn(g, x) { g.sqrt(x) }),
    ("sin", fn(g, x) { g.sin(x) }),
    ("cos", fn(g, x) { g.cos(x) }),
    ("tanh", fn(g, x) { g.tanh(x) }),
    ("sigmoid", fn(g, x) { g.sigmoid(x) }),
    ("softplus", fn(g, x) { g.softplus(x) }),
    ("silu", fn(g, x) { g.silu(x) }),
    ("gelu", fn(g, x) { g.gelu(x) }),
  ]
  for i = 0; i < scalar_cases.length(); i = i + 1 {
    let (name, build) = scalar_cases[i]
    let scalar = scalar_grad(build, point)
    let tensor = tensor_grad(
      fn(g, x) {
        match name {
          "neg" => g.neg(x)
          "exp" => g.exp(x)
          "ln" => g.ln(x)
          "sqrt" => g.sqrt(x)
          "sin" => g.sin(x)
          "cos" => g.cos(x)
          "tanh" => g.tanh(x)
          "sigmoid" => g.sigmoid(x)
          "softplus" => g.softplus(x)
          "silu" => g.silu(x)
          _ => g.gelu(x)
        }
      },
      point,
    )
    assert_close(
      tensor, scalar, 1.0e-12, "the \\{name} gradient should be the same in both engines",
    )
  }
}

///|
test "the_two_engines_agree_on_the_binary_operators" {
  let point = 0.7
  let other = 1.3
  // a + b, a - b, a * b and a / b, with b fixed.
  let scalar_cases : Array[(String, (@moongrad.Graph, Int32) -> Int32)] = [
    ("add", fn(g, x) { g.add(x, g.constant(other)) }),
    ("sub", fn(g, x) { g.sub(x, g.constant(other)) }),
    ("mul", fn(g, x) { g.mul(x, g.constant(other)) }),
    ("div", fn(g, x) { g.div(x, g.constant(other)) }),
  ]
  for i = 0; i < scalar_cases.length(); i = i + 1 {
    let (name, build) = scalar_cases[i]
    let scalar = scalar_grad(build, point)
    let tensor = tensor_grad(
      fn(g, x) {
        let k = g.constant(@moongrad.Tensor::scalar(other))
        match name {
          "add" => g.add(x, k)
          "sub" => g.sub(x, k)
          "mul" => g.mul(x, k)
          _ => g.div(x, k)
        }
      },
      point,
    )
    assert_close(
      tensor, scalar, 1.0e-12, "the \\{name} gradient should be the same in both engines",
    )
  }
}

///|
test "the_two_engines_agree_at_a_point_where_the_derivative_is_steep" {
  // The same check at a point where the derivatives are far from one:
  // a rule that lost a sign or a factor would not agree here either.
  let point = -2.4
  let scalar = scalar_grad(fn(g, x) { g.tanh(x) }, point)
  let tensor = tensor_grad(fn(g, x) { g.tanh(x) }, point)
  assert_close(tensor, scalar, 1.0e-12, "the tanh gradient at -2.4")
}

///|
test "the_two_engines_agree_on_a_chain" {
  // A chain exercises the rules in composition rather than one at a
  // time: a chain of exp, tanh and a product has to be differentiated
  // the same way on both sides.
  let point = 0.4
  let scalar = scalar_grad(
    fn(g, x) { g.mul(g.tanh(g.exp(x)), g.sin(x)) },
    point,
  )
  let tensor = tensor_grad(
    fn(g, x) { g.mul(g.tanh(g.exp(x)), g.sin(x)) },
    point,
  )
  assert_close(tensor, scalar, 1.0e-12, "the chained gradient")
}
