///|
/// The Hessian of the node `target` with respect to the input
/// tensors, evaluated at `b`.
///
/// Rows and columns are indexed by the *flattened* input space: the
/// elements of slot 0 come first, then those of slot 1, and so on.
/// A slot holding a shape `(2, 3)` tensor therefore occupies six
/// consecutive rows.
///
/// The construction is the scalar engine's, run one element at a
/// time. The first sweep compiled `d f / d x` into nodes; this picks
/// out the single element `d f / d x_{s,e}` and sweeps the gradient
/// graph again to compile `d^2 f / d x_j d x_{s,e}` out of it.
///
/// Nothing about the second sweep is special-cased. The gradient
/// graph is an ordinary tensor graph built from the same vocabulary
/// as the primal, and the rules that differentiate it are the same
/// rules that differentiate the primal. That is what the scalar
/// engine means by "the gradient is a program", and it is why the
/// tensor side gets second derivatives for free rather than by
/// growing a second reverse mode.
///
/// The target has to be a scalar. A tensor-valued objective has no
/// single Hessian; it has one per output element.
pub fn t_hessian(g : TGraph, b : TBinding, target : Int32) -> Matrix {
  let first = compile_tensor_gradient(g, b, target)
  let offsets = t_slot_offsets(b.inputs)
  let total = t_total_size(b.inputs)
  let h = Matrix::zeros(total, total)
  for s = 0; s < b.inputs.length(); s = s + 1 {
    let count = b.inputs[s].data.length()
    for e = 0; e < count; e = e + 1 {
      // The partial derivative with respect to this one input
      // element, as a node in the gradient graph.
      let element = first.graph.select(first.input_adj[s], e)
      let (rows, _) = t_second_sweep(first, b, element)
      for s2 = 0; s2 < b.inputs.length(); s2 = s2 + 1 {
        let block = rows[s2]
        for e2 = 0; e2 < block.data.length(); e2 = e2 + 1 {
          h.set(offsets[s2] + e2, offsets[s] + e, block.data[e2])
        }
      }
    }
  }
  h
}

///|
/// Where each slot's elements start in the flattened space.
fn t_slot_offsets(slots : Array[Tensor]) -> Array[Int32] {
  let offsets : Array[Int32] = []
  let mut total = 0
  for i = 0; i < slots.length(); i = i + 1 {
    offsets.push(total)
    total = total + slots[i].data.length()
  }
  offsets
}

///|
/// How many elements the flattened space holds.
fn t_total_size(slots : Array[Tensor]) -> Int32 {
  let mut total = 0
  for i = 0; i < slots.length(); i = i + 1 {
    total = total + slots[i].data.length()
  }
  total
}

///|
/// Differentiate one element of the gradient program once more.
///
/// The gradient graph is an ordinary tensor graph, so this is just
/// another backward sweep over it.
fn t_second_sweep(
  first : TGradGraph,
  b : TBinding,
  element : Int32,
) -> (Array[Tensor], Array[Tensor]) {
  let second = compile_tensor_gradient(first.graph, b, element)
  evaluate_tensor_gradient(second, b)
}

///|
/// The Hessian with respect to the parameters, evaluated at `b`.
///
/// Same construction as `t_hessian`, with the parameters in place of
/// the inputs. The flattened parameter space is the concatenation of
/// every parameter slot's elements.
pub fn t_param_hessian(g : TGraph, b : TBinding, target : Int32) -> Matrix {
  let first = compile_tensor_gradient(g, b, target)
  let offsets = t_slot_offsets(b.params)
  let total = t_total_size(b.params)
  let h = Matrix::zeros(total, total)
  for s = 0; s < b.params.length(); s = s + 1 {
    let count = b.params[s].data.length()
    for e = 0; e < count; e = e + 1 {
      let element = first.graph.select(first.param_adj[s], e)
      let (_, rows) = t_second_sweep(first, b, element)
      for s2 = 0; s2 < b.params.length(); s2 = s2 + 1 {
        let block = rows[s2]
        for e2 = 0; e2 < block.data.length(); e2 = e2 + 1 {
          h.set(offsets[s2] + e2, offsets[s] + e, block.data[e2])
        }
      }
    }
  }
  h
}

///|
/// The mixed Hessian: parameters against inputs.
///
/// Row `i` is the parameter element `i`, column `j` is the input
/// element `j`, and the entry is
/// `d^2 f / d w_i d x_j`. It is the block a Gauss-Newton step needs,
/// and it is the same construction as the other two: pick out one
/// element of the gradient and sweep the gradient graph again.
pub fn t_mixed_hessian(g : TGraph, b : TBinding, target : Int32) -> Matrix {
  let first = compile_tensor_gradient(g, b, target)
  let input_offsets = t_slot_offsets(b.inputs)
  let param_offsets = t_slot_offsets(b.params)
  let h = Matrix::zeros(t_total_size(b.params), t_total_size(b.inputs))
  for s = 0; s < b.inputs.length(); s = s + 1 {
    let count = b.inputs[s].data.length()
    for e = 0; e < count; e = e + 1 {
      let element = first.graph.select(first.input_adj[s], e)
      // The parameter half of the second sweep.
      let (_, prows) = t_second_sweep(first, b, element)
      for s2 = 0; s2 < b.params.length(); s2 = s2 + 1 {
        let block = prows[s2]
        for e2 = 0; e2 < block.data.length(); e2 = e2 + 1 {
          h.set(param_offsets[s2] + e2, input_offsets[s] + e, block.data[e2])
        }
      }
    }
  }
  h
}
