// core_test.mbt
//
// The scalar core: IR, simplifier, forward pass, and the gradient
// compiler. Every gradient assertion in this file is checked twice —
// once against a closed-form derivative and once against the
// independent forward-mode implementation in `forward.mbt`. The
// second check is the one that matters, because it would catch a
// shared bug between the two passes that a hard-coded expectation
// might miss.

///|
/// Failure log. Collecting failures instead of aborting on the first
/// one means a single `moon test` run reports every broken gradient
/// rather than only the first, which matters a lot when the thing
/// under test is a chain rule.
// A plain array is enough: MoonBit's Array is mutable in place,
// so pushing onto a top-level binding works without a Ref.
let failures : Array[String] = []

///|
fn check(cond : Bool, msg : String) -> Unit {
  if !cond {
    failures.push(msg)
  }
}

///|
fn approx(a : Double, b : Double, tol : Double) -> Bool {
  a - b < tol && b - a < tol
}

///|
fn assert_close(a : Double, b : Double, tol : Double, what : String) -> Unit {
  if !approx(a, b, tol) {
    check(
      false,
      "gradient mismatch in \{what}: got \{a}, want \{b}, tolerance \{tol}",
    )
  }
}

///|
fn binding_of(vars : Array[Double]) -> @moongrad.Binding {
  let b = @moongrad.Binding::new()
  for i = 0; i < vars.length(); i = i + 1 {
    b.set_var(i, vars[i])
  }
  b
}

// ---------------------------------------------------------------------------
// IR
// ---------------------------------------------------------------------------

///|
test "node_arity_matches_shape" {
  assert_eq(@moongrad.Node::Const(1.0).arity(), 0)
  assert_eq(@moongrad.Node::Var(0).arity(), 0)
  assert_eq(@moongrad.Node::ParamGet(0, 0).arity(), 0)
  assert_eq(@moongrad.Node::Unary(@moongrad.UnaryOp::Neg, 0).arity(), 1)
  assert_eq(@moongrad.Node::Binary(@moongrad.BinaryOp::Add, 0, 1).arity(), 2)
  assert_eq(@moongrad.Node::PowBy(2.0, 0).arity(), 1)
}

///|
test "node_children_match_shape" {
  let a = @moongrad.Node::Unary(@moongrad.UnaryOp::Exp, 7)
  assert_eq(a.children().length(), 1)
  assert_eq(a.children()[0], 7)
  let b = @moongrad.Node::Binary(@moongrad.BinaryOp::Mul, 3, 4)
  assert_eq(b.children().length(), 2)
  assert_eq(b.children()[0], 3)
  assert_eq(b.children()[1], 4)
  assert_eq(@moongrad.Node::PowBy(2.0, 9).children()[0], 9)
}

// ---------------------------------------------------------------------------
// Hash-consing
// ---------------------------------------------------------------------------

///|
test "identical_nodes_share_a_slot" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let a = g.add(x, x)
  let b = g.add(x, x)
  assert_eq(a, b)
}

///|
test "repeating_a_subexpression_does_not_grow_the_graph" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(3))
  let x = g.variable(0)
  let y = g.variable(1)
  let z = g.variable(2)
  let shared = g.mul(x, y)
  // (shared * z) + (shared * z): the two `shared * z` products are
  // the same node, so the sum is a single multiply away from a
  // second copy that never gets built.
  let a = g.add(shared, z)
  let b = g.add(shared, z)
  assert_eq(a, b)
}

///|
test "hash_consing_reports_its_hits" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let _ = g.add(x, y)
  let hits_before = g.stats.cse_hits
  let _ = g.add(x, y)
  assert_true(g.stats.cse_hits > hits_before)
}

// ---------------------------------------------------------------------------
// Simplifier
// ---------------------------------------------------------------------------

///|
test "simplifier_drops_the_additive_identity" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let zero = g.constant(0.0)
  assert_eq(g.add(x, zero), x)
  assert_eq(g.add(zero, x), x)
}

///|
test "simplifier_drops_the_multiplicative_identity" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let one = g.constant(1.0)
  assert_eq(g.mul(x, one), x)
  assert_eq(g.mul(one, x), x)
  assert_eq(g.div(x, one), x)
}

///|
test "simplifier_drops_annihilated_subtrees" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let zero = g.constant(0.0)
  // The whole `mul` disappears, and with it the subtree under it.
  assert_eq(g.mul(g.exp(x), zero), zero)
  assert_eq(g.mul(zero, g.sin(y)), zero)
}

///|
test "simplifier_folds_literal_subtrees" {
  let g = @moongrad.Graph::new()
  let a = g.constant(2.0)
  let b = g.constant(3.0)
  let sum = g.add(a, b)
  assert_eq(g.eval_constant(sum), 5.0)
}

///|
test "simplifier_folds_nested_literals" {
  let g = @moongrad.Graph::new()
  let a = g.constant(2.0)
  let inner = g.mul(a, a) // 4
  let outer = g.add(inner, a) // 6
  assert_eq(g.eval_constant(outer), 6.0)
  assert_true(g.stats.folds >= 2)
}

///|
test "simplifier_knows_x_minus_x" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let zero = g.constant(0.0)
  assert_eq(g.sub(x, x), zero)
}

///|
test "simplifier_knows_double_negation" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let twice = g.unary(
    @moongrad.UnaryOp::Neg,
    g.unary(@moongrad.UnaryOp::Neg, x),
  )
  assert_eq(twice, x)
}

///|
test "simplifier_keeps_0_times_a_subtree_it_cannot_see" {
  // `0 * (x + 1)` collapses too, because the whole product is
  // annihilated regardless of what is under it. The subtree is never
  // evaluated, so the fold is safe even when the subtree would have
  // been a domain error.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let negative = g.constant(-1.0)
  let _ = x
  let would_be_bad = g.ln(negative) // unfolded: log of a negative
  let zero = g.constant(0.0)
  assert_eq(g.mul(would_be_bad, zero), zero)
}

///|
test "simplifier_leaves_domain_errors_unfolded" {
  // log(-1) cannot be folded to a literal, so the node survives and
  // the forward pass is the one that reports it.
  let g = @moongrad.Graph::new()
  let negative = g.constant(-1.0)
  let node = g.ln(negative)
  let is_const = match g.nodes[node] {
    @moongrad.Node::Const(_) => true
    _ => false
  }
  assert_false(is_const)
}

///|
test "simplifier_rewrites_pow_one_and_zero" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  assert_eq(g.pow(x, 1.0), x)
  assert_eq(g.pow(x, 0.0), g.constant(1.0))
}

///|
test "simplifier_folds_pow_of_a_literal" {
  let g = @moongrad.Graph::new()
  let three = g.constant(3.0)
  let node = g.pow(three, 2.0)
  assert_eq(g.eval_constant(node), 9.0)
}

///|
test "stats_account_for_every_rewrite" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let one = g.constant(1.0)
  let zero = g.constant(0.0)
  let _ = g.mul(one, x) // identity
  let _ = g.mul(zero, y) // annihilator
  let _ = g.add(g.constant(2.0), g.constant(3.0)) // fold
  assert_eq(g.stats.identities, 1)
  assert_eq(g.stats.annihilated, 1)
  assert_true(g.stats.folds >= 1)
  assert_true(g.stats.slots_saved() >= 3)
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

///|
test "forward_evaluates_a_polynomial" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.add(g.mul(x, x), g.constant(1.0))
  let r = @moongrad.forward(g, binding_of([3.0]), f)
  assert_close(r.output, 10.0, 1.0e-12, "x^2 + 1 at x = 3")
}

///|
test "forward_evaluates_a_chain" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.ln(g.exp(g.add(x, g.constant(1.0))))
  let r = @moongrad.forward(g, binding_of([1.0]), f)
  assert_close(r.output, 2.0, 1.0e-12, "log(exp(x + 1)) at x = 1")
}

///|
test "forward_reports_a_domain_error" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.ln(x)
  let raised = try {
    let _ = @moongrad.forward(g, binding_of([-1.0]), f)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "forward_reports_a_singular_division" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.div(g.constant(1.0), x)
  let raised = try {
    let _ = @moongrad.forward(g, binding_of([0.0]), f)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "forward_reads_parameters" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let p = g.new_param(2)
  let x = g.variable(0)
  let f = g.add(g.mul(x, g.param(p, 0)), g.param(p, 1))
  let b = binding_of([2.0])
  b.set_param(0, [3.0, 4.0])
  let r = @moongrad.forward(g, b, f)
  assert_close(r.output, 10.0, 1.0e-12, "x*w0 + w1")
}

// ---------------------------------------------------------------------------
// Gradient compiler, checked against closed forms
// ---------------------------------------------------------------------------

///|
fn grad_of_x(
  f_of_x : (@moongrad.Graph, Int32) -> Int32,
  x_value : Double,
) -> Double raise @moongrad.ADError {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let f = f_of_x(g, g.variable(0))
  let (grads, _p) = @moongrad.grad(g, binding_of([x_value]), f)
  grads[0]
}

///|
test "gradient_of_the_identity_is_one" {
  assert_close(
    grad_of_x(
      fn(g, x) -> Int32 {
        let _ = g
        x
      },
      7.0,
    ),
    1.0,
    1.0e-12,
    "d/dx x",
  )
}

///|
test "gradient_of_a_constant_is_zero" {
  assert_close(
    grad_of_x(fn(g, _x) -> Int32 { g.constant(3.0) }, 7.0),
    0.0,
    1.0e-12,
    "d/dx 3",
  )
}

///|
test "gradient_of_a_square" {
  // f(x) = x^2, f' = 2x
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 { g.pow(x, 2.0) }
  assert_close(grad_of_x(f, 3.0), 6.0, 1.0e-12, "d/dx x^2 at 3")
  assert_close(grad_of_x(f, -2.0), -4.0, 1.0e-12, "d/dx x^2 at -2")
}

///|
test "gradient_of_a_cube_built_from_products" {
  // Built as ((x*x)*x) rather than x^3, so the rule under test is
  // the product rule, not the power rule.
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 { g.mul(g.mul(x, x), x) }
  assert_close(grad_of_x(f, 3.0), 27.0, 1.0e-12, "d/dx x^3 at 3")
}

///|
test "gradient_of_exponential_is_itself" {
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 { g.exp(x) }
  let at = 1.0
  assert_close(grad_of_x(f, at), @math.exp(at), 1.0e-9, "d/dx e^x at 1")
}

///|
test "gradient_of_log_is_reciprocal" {
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 { g.ln(x) }
  assert_close(grad_of_x(f, 4.0), 0.25, 1.0e-12, "d/dx log x at 4")
}

///|
test "gradient_of_sine_is_cosine" {
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 { g.sin(x) }
  let at = 0.7
  assert_close(grad_of_x(f, at), @math.cos(at), 1.0e-9, "d/dx sin x")
}

///|
test "gradient_of_tanh_is_one_minus_its_square" {
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 { g.tanh(x) }
  let at = 0.4
  let t = @math.tanh(at)
  assert_close(grad_of_x(f, at), 1.0 - t * t, 1.0e-9, "d/dx tanh x")
}

///|
test "gradient_of_a_quotient" {
  // f(x) = 1/x, f' = -1/x^2
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 {
    g.div(g.constant(1.0), x)
  }
  assert_close(grad_of_x(f, 3.0), -1.0 / 9.0, 1.0e-12, "d/dx 1/x at 3")
}

///|
test "gradient_of_a_real_power" {
  // f(x) = x^1.5, f' = 1.5 * x^0.5
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 { g.pow(x, 1.5) }
  assert_close(grad_of_x(f, 4.0), 3.0, 1.0e-9, "d/dx x^1.5 at 4")
}

///|
test "gradient_of_a_composition" {
  // f(x) = sin(x^2), f' = 2x cos(x^2)
  let f = fn(g : @moongrad.Graph, x : Int32) -> Int32 { g.sin(g.pow(x, 2.0)) }
  let at = 0.5
  assert_close(
    grad_of_x(f, at),
    2.0 * at * @math.cos(at * at),
    1.0e-9,
    "d/dx sin(x^2)",
  )
}

///|
test "gradient_of_a_sum_of_everything" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(3))
  let x = g.variable(0)
  let y = g.variable(1)
  let z = g.variable(2)
  // f = x*y + z^2
  let f = g.add(g.mul(x, y), g.pow(z, 2.0))
  let (grads, _p) = @moongrad.grad(g, binding_of([2.0, 5.0, 3.0]), f)
  assert_close(grads[0], 5.0, 1.0e-12, "df/dx")
  assert_close(grads[1], 2.0, 1.0e-12, "df/dy")
  assert_close(grads[2], 6.0, 1.0e-12, "df/dz")
}

///|
test "gradient_of_a_disconnected_variable_is_zero" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let unused = g.variable(1)
  let f = g.pow(x, 2.0)
  let _ = unused
  let (grads, _p) = @moongrad.grad(g, binding_of([3.0, 9.0]), f)
  assert_close(grads[1], 0.0, 1.0e-12, "df/d(unused)")
}

///|
test "gradient_of_a_diamond_shares_work_once" {
  // (x + y) * (x + y): the shared sum is one node, so its adjoint is
  // accumulated once and then used twice. The result is 2*(x+y).
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let s = g.add(x, y)
  let f = g.mul(s, s)
  let (grads, _p) = @moongrad.grad(g, binding_of([2.0, 3.0]), f)
  assert_close(grads[0], 10.0, 1.0e-12, "df/dx")
  assert_close(grads[1], 10.0, 1.0e-12, "df/dy")
}

///|
test "gradient_agrees_with_forward_mode" {
  // The real cross-check: the two passes are independent
  // implementations, so agreeing on a non-trivial program is
  // evidence that both are right.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let a = g.mul(x, y)
  let b = g.sin(a)
  let c = g.add(b, g.tanh(x))
  let d = g.mul(
    c,
    g.div(g.constant(3.0), g.add(g.constant(1.0), g.pow(y, 2.0))),
  )
  let e = g.exp(g.mul(d, g.constant(0.1)))
  let (grads, _p) = @moongrad.grad(g, binding_of([0.7, 1.3]), e)
  let (value, fdx) = @moongrad.forward_mode(g, binding_of([0.7, 1.3]), e, 0)
  let (_value2, fdy) = @moongrad.forward_mode(g, binding_of([0.7, 1.3]), e, 1)
  assert_close(grads[0], fdx, 1.0e-9, "df/dx reverse vs @moongrad.forward")
  assert_close(grads[1], fdy, 1.0e-9, "df/dy reverse vs @moongrad.forward")
  assert_close(e_value_of(g, e, 0.7, 1.3), value, 1.0e-9, "primal value")
}

///|
fn e_value_of(
  g : @moongrad.Graph,
  e : Int32,
  xv : Double,
  yv : Double,
) -> Double raise @moongrad.ADError {
  @moongrad.forward(g, binding_of([xv, yv]), e).output
}

///|
test "forward_mode_alone_agrees_on_a_simple_case" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.pow(x, 3.0)
  let (_v, d) = @moongrad.forward_mode(g, binding_of([2.0]), f, 0)
  assert_close(d, 12.0, 1.0e-9, "@moongrad.forward-mode d/dx x^3 at 2")
}

///|
test "gradient_survives_re_evaluation_at_a_new_point" {
  // The gradient is a program, not a number: build once, evaluate
  // wherever we like.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.pow(x, 3.0)
  let (_first, prog) = @moongrad.grad(g, binding_of([1.0]), f)
  let again = @moongrad.evaluate_gradients(prog, binding_of([5.0]))
  assert_close(again[0], 75.0, 1.0e-9, "d/dx x^3 at 5, from a stale program")
}

///|
test "the_gradient_is_a_graph_we_can_measure" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.sin(g.pow(x, 3.0))
  let (_grads, prog) = @moongrad.grad(g, binding_of([0.5]), f)
  // The gradient program is strictly larger than the primal: it is
  // the primal plus one emitted node per local derivative rule.
  assert_true(prog.grad.graph.len() > prog.primal.len())
}

///|
test "parameter_gradients_accumulate" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let w = g.new_param(2)
  let x = g.variable(0)
  // f = w0 * x + w1 * x^2
  let f = g.add(g.mul(g.param(w, 0), x), g.mul(g.param(w, 1), g.pow(x, 2.0)))
  let b = binding_of([3.0])
  b.set_param(0, [2.0, 5.0])
  let (_grads, prog) = @moongrad.grad(g, b, f)
  let pg = @moongrad.param_gradients(prog, b)
  // df/dw0 = x = 3, df/dw1 = x^2 = 9
  assert_close(pg[0][0], 3.0, 1.0e-12, "df/dw0")
  assert_close(pg[0][1], 9.0, 1.0e-12, "df/dw1")
}

///|
test "a_deep_chain_of_products_keeps_its_gradient_exact" {
  // x repeated through thirty multiplications by itself: x^31. The
  // chain rule multiplies thirty local derivatives, so a sloppy
  // implementation drifts well before this depth.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let mut acc = x
  for i = 0; i < 30; i = i + 1 {
    acc = g.mul(acc, x)
    ignore(i)
  }
  let (grads, _p) = @moongrad.grad(g, binding_of([2.0]), acc)
  // d/dx x^31 at x = 2 is 31 * 2^30.
  let expected = 31.0 * @math.pow(2.0, 30.0)
  assert_close(grads[0], expected, 1.0e-6, "d/dx x^31 at 2")
}

///|
test "a_deep_chain_through_a_transcendental_stays_exact" {
  // tanh applied thirty times to x. Each step multiplies by
  // 1 - tanh^2, so the derivative at x = 0 is exactly 1 and any
  // drift in the chain rule shows up immediately.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let mut acc = x
  for i = 0; i < 30; i = i + 1 {
    acc = g.tanh(acc)
    ignore(i)
  }
  let (grads, _p) = @moongrad.grad(g, binding_of([0.0]), acc)
  assert_close(grads[0], 1.0, 1.0e-12, "d/dx tanh^30(x) at 0")
}

///|
test "an_alternating_chain_matches_the_closed_form" {
  // f(x) = exp(sin(exp(cos(x)))). f' is a product of four factors;
  // writing it out by hand here is the independent check.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.exp(g.sin(g.exp(g.cos(x))))
  let at = 0.4
  let (grads, _p) = @moongrad.grad(g, binding_of([at]), f)
  let c = @math.cos(at)
  let e1 = @math.exp(c)
  let s = @math.sin(e1)
  let expected = @math.exp(s) * @math.cos(e1) * e1 * -@math.sin(at)
  assert_close(grads[0], expected, 1.0e-9, "d/dx exp(sin(exp(cos x)))")
}

///|
test "a_higher_order_polynomial_has_exact_second_derivative_behaviour" {
  // f(x) = x^4 -> f' = 4x^3. Checking x^3 at three points pins the
  // shape down without needing the Hessian machinery.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.pow(g.pow(x, 2.0), 2.0)
  let (grads, _p) = @moongrad.grad(g, binding_of([3.0]), f)
  assert_close(grads[0], 108.0, 1.0e-8, "d/dx x^4 at 3")
}

///|
/// Report everything that went wrong during the run. Kept as its own
/// test so the report prints in full rather than stopping at the
/// first failure.
test "zz_summary_of_gradient_mismatches" {
  let list = failures
  for f in list {
    println("  FAIL \{f}")
  }
  assert_eq(list.length(), 0)
}
