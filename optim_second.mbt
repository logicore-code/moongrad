// optim_second.mbt
//
// Second-order optimisation.
//
// The engine can now differentiate a gradient, so it can build a
// Hessian. Building the Hessian is not the interesting part, though:
// it is a dense matrix over the flattened parameter space, it costs
// one extra sweep per parameter element, and for anything but a small
// network it does not fit.
//
// What fits is a Hessian *vector product*. The direction is read from
// an input slot rather than baked into the program, so one
// compilation serves every direction, and a conjugate-gradient
// iteration can be run on the Hessian system without the Hessian
// ever being formed. That is Newton-CG, and it is the method that
// makes second-order optimisation usable at scale.
//
// Two products are provided, over the parameter space and over the
// input space. Both come out of the same construction, which is worth
// spelling out because it is the reason they are cheap.
//
// Let `g(x)` be the gradient. The Hessian is the transpose of the
// Jacobian of `g`, so
//
//     H v = grad( g . v )
//
// — the gradient of the scalar `g` dotted with `v`. So a
// Hessian-vector product is: build the scalar `g . v` out of the
// gradient program's own nodes, then differentiate that. The second
// derivative is not a special case anywhere; it is the ordinary
// backward sweep pointed at a different node.

///|
/// A compiled Hessian-vector product.
///
/// The direction is an input to the program rather than a constant
/// baked into it, so one compilation serves every direction. That
/// matters: a conjugate-gradient iteration needs a new direction at
/// every step, and re-compiling the product each time would cost
/// more than the iteration.
#derive(Debug)
pub(all) struct HvpProgram {
  /// The gradient program of the objective itself.
  first : TGradGraph
  /// The compiled product: a gradient program over the graph that
  /// holds the primal, the gradient, and the directional derivative.
  grad : TGradGraph
  /// The input slot the direction is read from.
  slot : Int32
  /// The number of elements the direction has.
  size : Int32
  /// The flat offset of each slot's first element.
  offsets : Array[Int32]
  /// True when the product is over the parameters, false when it is
  /// over the inputs.
  over_params : Bool
}

///|
pub extend HvpProgram with Debug::{to_repr}

///|
/// The flat size of a list of tensors, and where each one starts.
fn flat_layout(slots : Array[Tensor]) -> (Array[Int32], Int32) {
  let offsets : Array[Int32] = []
  let mut total = 0
  for i = 0; i < slots.length(); i = i + 1 {
    offsets.push(total)
    total = total + slots[i].data.length()
  }
  (offsets, total)
}

///|
/// Compile a Hessian-vector product.
///
/// The direction is read from input slot `slot`, which must not be
/// one of the primal's own input slots. The binding is given a slot
/// of the right size so that the graph and the binding agree.
///
/// The product is over the parameters when `over_params` is true and
/// over the inputs otherwise. Both are the same construction.
pub fn compile_hvp(
  g : TGraph,
  b : TBinding,
  target : Int32,
  slot : Int32,
  over_params : Bool,
) -> HvpProgram {
  // The space is snapshotted before the direction slot is added: over
  // the inputs, the direction is itself an input, and it must not be
  // part of the space it is dotted with.
  let space = if over_params { b.params.copy() } else { b.inputs.copy() }
  let (offsets, total) = flat_layout(space)
  // The direction is an input, so the binding has to have a slot for
  // it before the gradient program is compiled.
  b.set_input(slot, Tensor::zeros([total]))
  let first = compile_tensor_gradient(g, b, target)
  let h = first.graph
  let v = h.variable(slot)
  // q = sum over the space of (d f / d x) . v
  let adjoints = if over_params { first.param_adj } else { first.input_adj }
  let mut q = h.constant(Tensor::filled([], 0.0))
  for s = 0; s < space.length(); s = s + 1 {
    let count = space[s].data.length()
    if count == 0 {
      continue
    }
    let v_s = h.slice(v, 0, offsets[s], count)
    let g_s = h.reshape(adjoints[s], [count])
    let prod = h.mul(g_s, v_s)
    let dot = h.sum_axis(prod, 0)
    q = h.add(q, dot)
  }
  // The gradient of q is the Hessian-vector product.
  let second = compile_tensor_gradient(h, b, q)
  { first, grad: second, slot, size: total, offsets, over_params, }
}

///|
/// Compile a Hessian-vector product over the parameters.
pub fn compile_param_hvp(
  g : TGraph,
  b : TBinding,
  target : Int32,
  slot : Int32,
) -> HvpProgram {
  compile_hvp(g, b, target, slot, true)
}

///|
/// Compile a Hessian-vector product over the inputs.
pub fn compile_input_hvp(
  g : TGraph,
  b : TBinding,
  target : Int32,
  slot : Int32,
) -> HvpProgram {
  compile_hvp(g, b, target, slot, false)
}

///|
/// Apply a compiled Hessian-vector product.
///
/// The direction is written into the program's input slot and the
/// product is evaluated there. The binding is modified in place, so
/// the caller should not rely on that slot afterwards.
pub fn HvpProgram::apply(
  self : HvpProgram,
  b : TBinding,
  v : Array[Double],
) -> Array[Tensor] {
  b.set_input(self.slot, Tensor::vector(v))
  let values = t_forward(self.grad.graph, b)
  let out : Array[Tensor] = []
  if self.over_params {
    for i = 0; i < self.grad.param_adj.length(); i = i + 1 {
      out.push(values[self.grad.param_adj[i]])
    }
  } else {
    for i = 0; i < self.grad.input_adj.length(); i = i + 1 {
      out.push(values[self.grad.input_adj[i]])
    }
  }
  out
}

///|
/// A one-shot Hessian-vector product over the inputs.
pub fn t_input_hvp(
  g : TGraph,
  b : TBinding,
  target : Int32,
  v : Array[Double],
) -> Array[Tensor] {
  let hvp = compile_input_hvp(g, b, target, b.inputs.length())
  hvp.apply(b, v)
}

///|
/// A one-shot Hessian-vector product over the parameters.
pub fn t_param_hvp(
  g : TGraph,
  b : TBinding,
  target : Int32,
  v : Array[Double],
) -> Array[Tensor] {
  let hvp = compile_param_hvp(g, b, target, b.inputs.length())
  hvp.apply(b, v)
}

///|
/// The flattened direction that a list of parameter tensors
/// represents.
pub fn flatten_params(params : Array[Tensor]) -> Array[Double] {
  let out : Array[Double] = []
  for p in params {
    for x in p.data {
      out.push(x)
    }
  }
  out
}

///|
/// A matrix-free conjugate-gradient iteration.
///
/// The operator is given as a closure rather than a matrix, because
/// the whole point is to avoid building one. The operator must be
/// symmetric positive definite; the iteration returns `None` if it
/// finds a non-positive curvature, which is the usual symptom of an
/// indefinite operator.
pub fn cg_operator(
  op : (Array[Double]) -> Array[Double],
  b : Array[Double],
  x0 : Array[Double],
  max_iter : Int32,
  tol : Double,
) -> Array[Double]? {
  let n = b.length()
  let x = x0.copy()
  let mut r : Array[Double] = []
  for i = 0; i < n; i = i + 1 {
    r.push(b[i] - x[i])
  }
  let p = r.copy()
  let mut rr = dot(r, r)
  if rr.sqrt() < tol {
    return Some(x)
  }
  let mut iter = 0
  while iter < max_iter {
    iter = iter + 1
    let ap = op(p)
    let pap = dot(p, ap)
    if pap <= 0.0 {
      return None
    }
    let alpha = rr / pap
    for i = 0; i < n; i = i + 1 {
      x[i] = x[i] + alpha * p[i]
      r[i] = r[i] - alpha * ap[i]
    }
    // The recursive residual drifts, so it is recomputed.
    let ax = op(x)
    let r2 : Array[Double] = []
    for i = 0; i < n; i = i + 1 {
      r2.push(b[i] - ax[i])
    }
    let rr_new = dot(r2, r2)
    if rr_new.sqrt() < tol {
      return Some(x)
    }
    let beta = rr_new / rr
    for i = 0; i < n; i = i + 1 {
      p[i] = r2[i] + beta * p[i]
    }
    r = r2
    rr = rr_new
  }
  Some(x)
}

///|
/// The inner product of two vectors.
pub fn dot(a : Array[Double], b : Array[Double]) -> Double {
  let n = if a.length() < b.length() { a.length() } else { b.length() }
  let mut acc = 0.0
  for i = 0; i < n; i = i + 1 {
    acc = acc + a[i] * b[i]
  }
  acc
}

///|
/// A Newton step, computed by conjugate gradients on the damped
/// Hessian system.
///
/// The system solved is
///
///     (H + damping I) dx = -g
///
/// where `H` is the Hessian over the parameter space and `g` is the
/// gradient. The damping is what makes the system positive definite:
/// the exact Hessian of a non-convex objective is often indefinite,
/// and conjugate gradients needs a positive definite operator. This
/// is Levenberg-Marquardt's answer to the same problem, applied to
/// the exact Hessian rather than to the Gauss-Newton approximation.
///
/// The Hessian is never formed. Every application of the operator is
/// one Hessian-vector product, and the whole iteration is a single
/// compiled program re-evaluated at a new direction each step.
///
/// Returns `None` when the iteration breaks down, which means the
/// damped system was still not positive definite.
pub fn newton_cg_step(
  g : TGraph,
  b : TBinding,
  target : Int32,
  damping : Double,
  max_iter : Int32,
  tol : Double,
  slot : Int32,
) -> Array[Double]? {
  let hvp = compile_param_hvp(g, b, target, slot)
  // The gradient of the objective, read out of the gradient program
  // the Hessian-vector product was compiled from.
  let first = hvp.first
  let gvals = t_forward(first.graph, b)
  let grad_flat : Array[Double] = []
  for s = 0; s < b.params.length(); s = s + 1 {
    let t = gvals[first.param_adj[s]]
    for x in t.data {
      grad_flat.push(x)
    }
  }
  let rhs : Array[Double] = []
  for x in grad_flat {
    rhs.push(-x)
  }
  let op = fn(v : Array[Double]) -> Array[Double] {
    let blocks = hvp.apply(b, v)
    let out : Array[Double] = []
    for t in blocks {
      for x in t.data {
        out.push(x)
      }
    }
    for i = 0; i < out.length(); i = i + 1 {
      out[i] = out[i] + damping * v[i]
    }
    out
  }
  let zeros : Array[Double] = []
  for i = 0; i < rhs.length(); i = i + 1 {
    zeros.push(0.0)
  }
  cg_operator(op, rhs, zeros, max_iter, tol)
}

///|
/// A damped Newton step computed from the dense Hessian.
///
/// The dense route is here for comparison and for the small problems
/// where the Hessian does fit: it solves `(H + damping I) dx = -g`
/// with a Cholesky factorisation, which is exact rather than
/// iterative. `newton_cg_step` is the same step without the matrix.
pub fn dense_newton_step(
  g : TGraph,
  b : TBinding,
  target : Int32,
  damping : Double,
) -> Array[Double]? {
  let h = t_param_hessian(g, b, target)
  let first = compile_tensor_gradient(g, b, target)
  let gvals = t_forward(first.graph, b)
  let grad_flat : Array[Double] = []
  for s = 0; s < b.params.length(); s = s + 1 {
    let t = gvals[first.param_adj[s]]
    for x in t.data {
      grad_flat.push(x)
    }
  }
  let n = h.nrows()
  let damped = h.add(Matrix::identity(n).scale(damping))
  let rhs : Array[Double] = []
  for x in grad_flat {
    rhs.push(-x)
  }
  cholesky_solve(damped, rhs)
}
