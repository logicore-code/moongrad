// tensor_grad.mbt
//
// Reverse-mode automatic differentiation over tensors.
//
// The shape of the design is the same as the scalar engine in
// `reverse.mbt`, with one extra idea: a graph node here stands for
// a whole tensor, not a single number. The forward pass produces
// one tensor per node; the backward pass produces one *adjoint
// tensor* per node.
//
// What makes it work is the same index-ordering invariant the
// scalar engine relies on: a node can only be built after its
// children exist, so every edge points from a higher slot to a
// lower one. The forward pass is a left-to-right scan and the
// backward pass a right-to-left one. Neither needs a topological
// sort.
//
// The one genuinely new piece is broadcasting. When two operands
// of different shapes are combined, the result has the broadcast
// shape, and the gradient that comes back is therefore *larger*
// than the gradient of the smaller operand. It has to be summed
// back down to the operand's shape. `t_sum_to_shape` does that,
// and it is the single function that makes every broadcasting
// operator's backward rule correct.

///|
/// A node of a tensor computation.
pub(all) enum TNode {
  /// A constant, held in the graph's constant pool.
  Const(Int)
  /// An input tensor, identified by slot.
  Var(Int)
  /// A read of a whole parameter vector.
  ParamRead(Int)
  /// An elementwise unary operator.
  Unary(UnaryOp, Int)
  /// An elementwise binary operator, with broadcasting.
  Binary(BinaryOp, Int, Int)
  /// Matrix product.
  MatMul(Int, Int)
  /// Sum along one axis, dropping that axis.
  Sum(Int, Int)
  /// Reshape to a new shape.
  Reshape(Int, Array[Int])
  /// Permute the axes.
  Transpose(Int, Array[Int])
  /// Concatenate two tensors along an axis.
  Concat(Int, Int, Int)
  /// A slice along one axis.
  Slice(Int, Int, Int, Int)
  /// Multiply by a constant tensor held in the pool.
  ScaleBy(Int, Int)
  /// Read one row of a parameter vector.
  GatherRow(Int, Int, Int)
  /// `base ^ exponent`, with the exponent carried inline.
  PowBy(Double, Int)
  /// A 2-D convolution. The operands are the input and the weight;
  /// the rest are the geometry.
  Conv2d(Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)
} derive(Debug)

///|
/// A tensor computation under construction.
///
/// Unlike the scalar engine this one does not hash-cons: a tensor
/// node stands for a whole array, so sharing it would save very
/// little, and the index map it would need makes the node
/// unhashable. The scalar engine keeps its own graph for the cases
/// where the sharing actually pays.
pub(all) struct TGraph {
  mut nodes : Array[TNode]
  mut pool : Array[Tensor]
  mut values : Array[Tensor]
  /// The shape of every node's value, filled by the forward pass.
  mut shapes : Array[Array[Int]]
  /// Set when an operation could not be evaluated. A silent
  /// degradation to a zero scalar would turn a shape mistake into a
  /// wrong gradient, which is much harder to find than an error.
  mut failed : Bool
  /// How many parameter slots have been handed out.
  mut param_count : Int
} derive(Debug)

///|
/// An empty tensor graph.
pub fn TGraph::new() -> TGraph {
  {
    nodes: [],
    pool: [],
    values: [],
    shapes: [],
    failed: false,
    param_count: 0,
  }
}

///|
/// The number of nodes.
pub fn TGraph::len(self : TGraph) -> Int {
  self.nodes.length()
}

///|
/// Add a constant to the pool and return the *node* that reads it.
///
/// The distinction matters: the pool slot identifies the value, but
/// a node slot identifies a position in the graph. Returning the
/// pool index here would make every constant in a graph alias
/// whichever node happened to sit at that position, which produces a
/// program that runs and computes the wrong thing.
pub fn TGraph::constant(self : TGraph, t : Tensor) -> Int {
  let slot = self.pool.length()
  self.pool.push(t)
  self.nodes.push(TNode::Const(slot))
  self.nodes.length() - 1
}

///|
/// An input tensor slot, by name of the slot.
pub fn TGraph::variable(self : TGraph, slot : Int) -> Int {
  self.input(slot)
}

///|
/// An input tensor slot.
pub fn TGraph::input(self : TGraph, slot : Int) -> Int {
  self.nodes.push(TNode::Var(slot))
  self.nodes.length() - 1
}

///|
/// A parameter vector slot.
pub fn TGraph::param(self : TGraph, slot : Int) -> Int {
  self.nodes.push(TNode::ParamRead(slot))
  self.nodes.length() - 1
}

///|
/// Allocate a fresh parameter slot and return its index.
///
/// Slots are allocated, never sized. A layer that wants a parameter
/// of a hundred elements still gets one slot; the number of
/// elements is the size of the tensor that fills it.
pub fn TGraph::new_param_slot(self : TGraph) -> Int {
  let slot = self.param_count
  self.param_count = self.param_count + 1
  slot
}

///|
/// An elementwise unary operator.
pub fn TGraph::unary(self : TGraph, op : UnaryOp, child : Int) -> Int {
  self.nodes.push(TNode::Unary(op, child))
  self.nodes.length() - 1
}

///|
/// An elementwise binary operator, with broadcasting.
pub fn TGraph::binary(self : TGraph, op : BinaryOp, a : Int, b : Int) -> Int {
  self.nodes.push(TNode::Binary(op, a, b))
  self.nodes.length() - 1
}

///|
/// Matrix product.
pub fn TGraph::matmul(self : TGraph, a : Int, b : Int) -> Int {
  self.nodes.push(TNode::MatMul(a, b))
  self.nodes.length() - 1
}

///|
/// Sum along an axis, dropping that axis.
pub fn TGraph::sum_axis(self : TGraph, child : Int, axis : Int) -> Int {
  self.nodes.push(TNode::Sum(child, axis))
  self.nodes.length() - 1
}

///|
/// Reshape to a new shape.
pub fn TGraph::reshape(self : TGraph, child : Int, shape : Array[Int]) -> Int {
  self.nodes.push(TNode::Reshape(child, shape.copy()))
  self.nodes.length() - 1
}

///|
/// Permute the axes.
pub fn TGraph::transpose(self : TGraph, child : Int, perm : Array[Int]) -> Int {
  self.nodes.push(TNode::Transpose(child, perm.copy()))
  self.nodes.length() - 1
}

///|
/// Concatenate along an axis.
pub fn TGraph::concat(self : TGraph, a : Int, b : Int, axis : Int) -> Int {
  self.nodes.push(TNode::Concat(a, b, axis))
  self.nodes.length() - 1
}

///|
/// A slice along an axis.
pub fn TGraph::slice(
  self : TGraph,
  child : Int,
  axis : Int,
  start : Int,
  count : Int,
) -> Int {
  self.nodes.push(TNode::Slice(child, axis, start, count))
  self.nodes.length() - 1
}

///|
/// Multiply a node by a constant tensor.
pub fn TGraph::scale_tensor(self : TGraph, x : Int, k : Tensor) -> Int {
  // The constant goes into the pool; the node refers to it by pool
  // slot, which is a different numbering from the node slots.
  let slot = self.pool.length()
  self.pool.push(k)
  self.nodes.push(TNode::ScaleBy(x, slot))
  self.nodes.length() - 1
}

///|
/// Read one row of a parameter vector.
pub fn TGraph::gather_row(
  self : TGraph,
  param_slot : Int,
  index : Int,
  dim : Int,
) -> Int {
  self.nodes.push(TNode::GatherRow(param_slot, index, dim))
  self.nodes.length() - 1
}

///|
/// The shape of a node's value, as recorded by the last forward
/// pass. Returns an empty shape for a node that has not been
/// evaluated.
pub fn TGraph::last_shape(self : TGraph, x : Int) -> Array[Int] {
  if x >= 0 && x < self.shapes.length() {
    self.shapes[x].copy()
  } else {
    []
  }
}

///|
/// Elementwise helpers, so that a layer reads like arithmetic
/// rather than like graph construction.

///|
pub fn TGraph::pow(self : TGraph, x : Int, exponent : Double) -> Int {
  self.nodes.push(TNode::PowBy(exponent, x))
  self.nodes.length() - 1
}

///|
pub fn TGraph::add(self : TGraph, a : Int, b : Int) -> Int {
  self.binary(BinaryOp::Add, a, b)
}

///|
pub fn TGraph::sub(self : TGraph, a : Int, b : Int) -> Int {
  self.binary(BinaryOp::Sub, a, b)
}

///|
pub fn TGraph::mul(self : TGraph, a : Int, b : Int) -> Int {
  self.binary(BinaryOp::Mul, a, b)
}

///|
pub fn TGraph::div(self : TGraph, a : Int, b : Int) -> Int {
  self.binary(BinaryOp::Div, a, b)
}

///|
pub fn TGraph::neg(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Neg, x)
}

///|
pub fn TGraph::exp(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Exp, x)
}

///|
pub fn TGraph::ln(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Ln, x)
}

///|
pub fn TGraph::sqrt(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Sqrt, x)
}

///|
pub fn TGraph::sin(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Sin, x)
}

///|
pub fn TGraph::cos(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Cos, x)
}

///|
pub fn TGraph::tanh(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Tanh, x)
}

///|
pub fn TGraph::sigmoid(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Sigmoid, x)
}

///|
pub fn TGraph::softplus(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Softplus, x)
}

///|
pub fn TGraph::relu(self : TGraph, x : Int) -> Int {
  self.unary(UnaryOp::Relu, x)
}

///|
/// A 2-D convolution node.
pub fn TGraph::conv2d(
  self : TGraph,
  input : Int,
  weight : Int,
  in_channels : Int,
  out_channels : Int,
  kernel_h : Int,
  kernel_w : Int,
  stride_h : Int,
  stride_w : Int,
  pad_h : Int,
  pad_w : Int,
) -> Int {
  self.nodes.push(
    TNode::Conv2d(
      input, weight, in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w,
      pad_h, pad_w,
    ),
  )
  self.nodes.length() - 1
}

///|
/// The inputs and parameters a tensor graph is evaluated against.
pub(all) struct TBinding {
  /// Value of each input tensor, indexed by slot.
  inputs : Array[Tensor]
  /// Value of each parameter vector, indexed by slot.
  params : Array[Tensor]
} derive(Debug)

///|
/// An empty binding.
pub fn TBinding::new() -> TBinding {
  { inputs: [], params: [], }
}

///|
/// Set the value of an input tensor.
pub fn TBinding::set_input(self : TBinding, slot : Int, value : Tensor) -> Unit {
  while self.inputs.length() <= slot {
    self.inputs.push(Tensor::zeros([]))
  }
  self.inputs[slot] = value
}

///|
/// Set the value of a parameter vector.
pub fn TBinding::set_param(self : TBinding, slot : Int, value : Tensor) -> Unit {
  while self.params.length() <= slot {
    self.params.push(Tensor::zeros([]))
  }
  self.params[slot] = value
}

///|
/// Evaluate every node of `g` under `b` and return the value array.
pub fn t_forward(g : TGraph, b : TBinding) -> Array[Tensor] {
  let n = g.nodes.length()
  g.failed = false
  let values : Array[Tensor] = []
  for i = 0; i < n; i = i + 1 {
    values.push(Tensor::zeros([]))
  }
  for i = 0; i < n; i = i + 1 {
    values[i] = t_eval_node(g, b, values, i)
  }
  // Record the shapes so a layer can ask for a node's shape
  // without re-running the forward pass.
  while g.shapes.length() < n {
    g.shapes.push([])
  }
  for i = 0; i < n; i = i + 1 {
    g.shapes[i] = values[i].shape.copy()
  }
  values
}

///|
/// Evaluate the graph and report whether every operation succeeded.
///
/// A `matmul` whose shapes do not line up, or a `reshape` that
/// changes the element count, used to fall back to a zero scalar.
/// That is the worst possible failure: the graph still runs, the
/// loss still looks finite, and the gradient is silently wrong.
pub fn t_forward_checked(g : TGraph, b : TBinding) -> Array[Tensor]? {
  let values = t_forward(g, b)
  if g.failed {
    None
  } else {
    Some(values)
  }
}

///|
fn t_eval_node(
  g : TGraph,
  b : TBinding,
  values : Array[Tensor],
  id : Int,
) -> Tensor {
  match g.nodes[id] {
    TNode::Const(slot) => g.pool[slot]
    TNode::Var(slot) =>
      if slot >= 0 && slot < b.inputs.length() {
        b.inputs[slot]
      } else {
        Tensor::zeros([])
      }
    TNode::ParamRead(slot) =>
      if slot >= 0 && slot < b.params.length() {
        b.params[slot]
      } else {
        Tensor::zeros([])
      }
    TNode::Unary(op, child) => t_unary(op, values[child])
    TNode::Binary(op, a, c) =>
      match t_binary(op, values[a], values[c]) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::MatMul(a, c) =>
      match values[a].matmul(values[c]) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::Sum(child, axis) =>
      match t_sum_axis(values[child], axis) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::Reshape(child, shape) =>
      match values[child].reshape(shape) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::Transpose(child, perm) =>
      match values[child].transpose(perm) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::Concat(a, c, axis) =>
      match values[a].concat(values[c], axis) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::Slice(child, axis, start, count) =>
      match values[child].slice(axis, start, count) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::ScaleBy(child, slot) => t_mul_or(values[child], g.pool[slot])
    TNode::PowBy(exponent, child) => t_pow(values[child], exponent)
    TNode::Conv2d(
      input,
      weight,
      in_channels,
      out_channels,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w
    ) =>
      match
        t_conv2d(
          values[input],
          values[weight],
          None,
          stride_h,
          stride_w,
          pad_h,
          pad_w,
        ) {
        Some(t) => t
        None => Tensor::zeros([])
      }
    TNode::GatherRow(param_slot, index, dim) => {
      let row = Tensor::zeros([dim])
      if param_slot >= 0 && param_slot < b.params.length() {
        let table = b.params[param_slot]
        for i = 0; i < dim; i = i + 1 {
          let src = index * dim + i
          if src >= 0 && src < table.data.length() {
            row.data[i] = table.data[src]
          }
        }
      }
      row
    }
  }
}

///|
/// The result of a backward pass.
pub(all) struct TGrad {
  /// Gradient with respect to each input tensor.
  input_grads : Array[Tensor]
  /// Gradient with respect to each parameter vector.
  param_grads : Array[Tensor]
  /// Value of the differentiated node.
  output : Tensor
} derive(Debug)

///|
/// Differentiate `target` with respect to every input and every
/// parameter.
///
/// The backward pass is the scalar engine's sweep with tensors
/// instead of numbers, plus the broadcasting adjustment: when a
/// child was broadcast up to the parent's shape, its gradient comes
/// back larger and has to be summed down to the child's own shape.
pub fn t_grad(g : TGraph, b : TBinding, target : Int) -> TGrad {
  let values = t_forward(g, b)
  let n = g.nodes.length()
  // Adjoints, one tensor per node.
  let adj : Array[Tensor] = []
  for i = 0; i < n; i = i + 1 {
    adj.push(Tensor::zeros([]))
  }
  let input_grads : Array[Tensor] = []
  for i = 0; i < b.inputs.length(); i = i + 1 {
    input_grads.push(Tensor::zeros([]))
  }
  let param_grads : Array[Tensor] = []
  for i = 0; i < b.params.length(); i = i + 1 {
    param_grads.push(Tensor::zeros(b.params[i].shape))
  }
  if target < 0 || target >= n {
    return { input_grads, param_grads, output: Tensor::zeros([]), }
  }
  // The adjoint of the differentiated node is a tensor of ones:
  // every element of the output contributes to the derivative.
  adj[target] = Tensor::filled(values[target].shape, 1.0)
  for i = n - 1; i >= 0; i = i - 1 {
    if t_is_zero(adj[i]) {
      continue
    }
    t_push_adjoints(g, values, adj, input_grads, param_grads, i)
  }
  { input_grads, param_grads, output: values[target], }
}

///|
/// `true` when every entry of the tensor is zero.
pub fn t_is_zero(t : Tensor) -> Bool {
  for x in t.data {
    if x != 0.0 {
      return false
    }
  }
  true
}

///|
/// Sum a tensor down to `target`, over the axes where it was
/// broadcast up.
///
/// This is the function that makes broadcasting work in reverse
/// mode. If `a` has shape `(3,)` and `b` has shape `(2, 3)`, then
/// `a` was repeated along the first axis, and the gradient that
/// comes back for `a` has shape `(2, 3)`. The gradient with
/// respect to `a` is the sum of that tensor over the first axis.
///
/// The target's shape is the operand's own shape, and the incoming
/// tensor's shape is the broadcast shape. Aligning the two from the
/// right gives the axes to sum:
///
///   * axes the target does not have at all — the operand was
///     broadcast along a whole new axis;
///   * axes where the target has length one and the broadcast shape
///     has more — the operand's single element was repeated.
///
/// The result has exactly the target's shape, which is what the
/// caller needs to add it into the operand's adjoint.
pub fn t_sum_to_shape(t : Tensor, target : Array[Int]) -> Tensor {
  let extra = t.shape.length() - target.length()
  // Axes the target does not have are summed away entirely.
  let mut acc = t
  for d = extra - 1; d >= 0; d = d - 1 {
    match t_sum_axis(acc, d) {
      Some(x) => acc = x
      None => ()
    }
  }
  // Axes where the target has length one but the broadcast shape has
  // more are summed but kept, with length one.
  for d = 0; d < acc.shape.length(); d = d + 1 {
    if d < target.length() && target[d] == 1 && acc.shape[d] != 1 {
      match t_sum_axis_keep(acc, d) {
        Some(x) => acc = x
        None => ()
      }
    }
  }
  acc
}

///|
/// Sum along an axis but *keep* the axis, so the result has the same
/// rank and the axis has length one.
///
/// The output's buffer is sized for the reduced shape, so the
/// indexing uses the reduced strides. Writing it with the original
/// strides would read and write the wrong elements as soon as the
/// axis is longer than one.
pub fn t_sum_axis_keep(t : Tensor, axis : Int) -> Tensor? {
  if axis < 0 || axis >= t.shape.length() {
    return None
  }
  let r = t.shape.length()
  let mut outer = 1
  for d = 0; d < axis; d = d + 1 {
    outer = outer * t.shape[d]
  }
  let mut inner = 1
  for d = axis + 1; d < r; d = d + 1 {
    inner = inner * t.shape[d]
  }
  let len = t.shape[axis]
  let out_shape = t.shape.copy()
  out_shape[axis] = 1
  let out = Tensor::zeros(out_shape)
  for o = 0; o < outer; o = o + 1 {
    for i = 0; i < len; i = i + 1 {
      for j = 0; j < inner; j = j + 1 {
        let src = o * len * inner + i * inner + j
        let dst = o * inner + j
        out.data[dst] = out.data[dst] + t.data[src]
      }
    }
  }
  Some(out)
}

///|
/// Push the adjoint of node `i` into its children.
fn t_push_adjoints(
  g : TGraph,
  values : Array[Tensor],
  adj : Array[Tensor],
  input_grads : Array[Tensor],
  param_grads : Array[Tensor],
  i : Int,
) -> Unit {
  let gbar = adj[i]
  match g.nodes[i] {
    TNode::Const(_) => ()
    TNode::Var(slot) =>
      if slot >= 0 && slot < input_grads.length() {
        input_grads[slot] = t_add_or(input_grads[slot], gbar)
      }
    TNode::ParamRead(slot) =>
      if slot >= 0 && slot < param_grads.length() {
        param_grads[slot] = t_add_or(param_grads[slot], gbar)
      }
    TNode::Unary(op, child) => {
      let local = t_unary_derivative(op, values[child])
      let contrib = t_mul_or(gbar, local)
      adj[child] = t_add_or(adj[child], contrib)
    }
    TNode::Binary(op, a, c) => {
      let va = values[a]
      let vc = values[c]
      // The two local derivatives are evaluated in the broadcast
      // shape and then summed back down to each operand's shape.
      let common = match common_shape(va.shape, vc.shape) {
        Some(s) => s
        None => return
      }
      let da = t_binary_dleft(op, va, vc, common)
      let db = t_binary_dright(op, va, vc, common)
      let contrib_a = t_mul_or(gbar, da)
      let contrib_b = t_mul_or(gbar, db)
      adj[a] = t_add_or(adj[a], t_sum_to_shape(contrib_a, va.shape))
      adj[c] = t_add_or(adj[c], t_sum_to_shape(contrib_b, vc.shape))
    }
    TNode::MatMul(a, c) => {
      let ga = match gbar.matmul_t(values[c]) {
        Some(t) => t
        None => return
      }
      let gb = match values[a].transpose([1, 0]) {
        Some(at) =>
          match at.matmul(gbar) {
            Some(t) => t
            None => return
          }
        None => return
      }
      adj[a] = t_add_or(adj[a], ga)
      adj[c] = t_add_or(adj[c], gb)
    }
    TNode::Sum(child, axis) => {
      // d(sum_axis(x))/dx is the gradient broadcast back up to x's
      // shape, with the summed axis re-inserted as a size-1 axis.
      let contrib = t_expand_axis(gbar, axis, values[child].shape)
      adj[child] = t_add_or(adj[child], contrib)
    }
    TNode::Reshape(child, _) => {
      let contrib = match gbar.reshape(values[child].shape) {
        Some(t) => t
        None => return
      }
      adj[child] = t_add_or(adj[child], contrib)
    }
    TNode::Transpose(child, perm) => {
      let inverse = t_inverse_perm(perm)
      let contrib = match gbar.transpose(inverse) {
        Some(t) => t
        None => return
      }
      adj[child] = t_add_or(adj[child], contrib)
    }
    TNode::Concat(a, c, axis) => {
      let la = values[a].shape[axis]
      let lc = values[c].shape[axis]
      let pa = match gbar.slice(axis, 0, la) {
        Some(t) => t
        None => return
      }
      let pc = match gbar.slice(axis, la, lc) {
        Some(t) => t
        None => return
      }
      adj[a] = t_add_or(adj[a], pa)
      adj[c] = t_add_or(adj[c], pc)
    }
    TNode::ScaleBy(child, slot) => {
      // d/dx = gbar * k, with k the same constant.
      let contrib = t_mul_or(gbar, g.pool[slot])
      adj[child] = t_add_or(adj[child], contrib)
    }
    TNode::PowBy(exponent, child) => {
      // d/dx x^c = c * x^(c-1)
      let x = values[child]
      let mut d = Tensor::zeros(x.shape)
      if x.data[0] != 0.0 {
        d = t_scale(t_pow(x, exponent - 1.0), exponent)
      }
      let contrib = t_mul_or(gbar, d)
      adj[child] = t_add_or(adj[child], contrib)
    }
    TNode::Conv2d(
      input,
      weight,
      in_channels,
      out_channels,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w
    ) => {
      let xi = values[input]
      let wi = values[weight]
      if xi.shape.length() != 4 || wi.shape.length() != 4 {
        return
      }
      let batch = xi.shape[0]
      let height = xi.shape[2]
      let width = xi.shape[3]
      let out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1
      let out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1
      if out_h <= 0 || out_w <= 0 {
        return
      }
      let row_len = in_channels * kernel_h * kernel_w
      // The incoming gradient is (batch, out_ch, out_h, out_w); the
      // matrix form wants (batch*out_h*out_w, out_ch).
      let gbar_t = match gbar.transpose([0, 2, 3, 1]) {
        Some(t) => t
        None => return
      }
      let gbar_rows = match
        gbar_t.reshape([batch * out_h * out_w, out_channels]) {
        Some(t) => t
        None => return
      }
      // The weight flattened to (out_ch, row_len).
      let w_flat = Tensor::zeros([out_channels, row_len])
      for oc = 0; oc < out_channels; oc = oc + 1 {
        for c = 0; c < in_channels; c = c + 1 {
          for kh = 0; kh < kernel_h; kh = kh + 1 {
            for kw = 0; kw < kernel_w; kw = kw + 1 {
              let src = ((oc * in_channels + c) * kernel_h + kh) * kernel_w + kw
              let dst = oc * row_len + (c * kernel_h + kh) * kernel_w + kw
              w_flat.data[dst] = wi.data[src]
            }
          }
        }
      }
      // d/d(patches) = d/out @ w
      let d_patches = match gbar_rows.matmul(w_flat) {
        Some(t) => t
        None => return
      }
      // d/d(weight) = patches^T @ d/out
      let padded = t_pad2d(xi, pad_h, pad_w)
      let patches = t_im2col(
        padded, out_h, out_w, in_channels, kernel_h, kernel_w, stride_h, stride_w,
      )
      let d_w_rows = match patches.transpose([1, 0]) {
        Some(t) => t
        None => return
      }
      let d_w_rows = match d_w_rows.matmul(gbar_rows) {
        Some(t) => t
        None => return
      }
      let d_w = match d_w_rows.transpose([1, 0]) {
        Some(t) => t
        None => return
      }
      let d_w = match
        d_w.reshape([out_channels, in_channels, kernel_h, kernel_w]) {
        Some(t) => t
        None => return
      }
      // d/d(input) = col2im(d/d(patches)), cropped back out of the
      // padding.
      let d_padded = t_col2im(
        d_patches,
        batch,
        height + 2 * pad_h,
        width + 2 * pad_w,
        out_h,
        out_w,
        in_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
      )
      let d_input = t_crop2d(d_padded, pad_h, pad_w)
      adj[input] = t_add_or(adj[input], d_input)
      // The weight's gradient goes to the parameter slot that holds
      // it. The node only knows the weight's node id, so the
      // gradient is parked on that node and picked up when the
      // parameter read is reached.
      adj[weight] = t_add_or(adj[weight], d_w)
    }
    TNode::GatherRow(param_slot, index, dim) =>
      // The gradient of a gather is a scatter: one row of the
      // parameter's gradient gets the incoming gradient.
      if param_slot >= 0 && param_slot < param_grads.length() {
        let full = param_grads[param_slot]
        for i = 0; i < dim; i = i + 1 {
          let dst = index * dim + i
          if dst >= 0 && dst < full.data.length() {
            full.data[dst] = full.data[dst] + gbar.data[i]
          }
        }
        // The parameter's gradient may have been empty before; make
        // sure it has the right size now.
        param_grads[param_slot] = full
      }
    TNode::Slice(child, axis, start, count) => {
      // The gradient of a slice is zero everywhere except in the
      // slice itself, where it is the incoming gradient.
      let full = Tensor::zeros(values[child].shape)
      let part = match gbar.slice(axis, 0, count) {
        Some(t) => t
        None => return
      }
      // Copy `part` into `full` at the right offset.
      let r = full.shape.length()
      let mut outer = 1
      for d = 0; d < axis; d = d + 1 {
        outer = outer * full.shape[d]
      }
      let mut inner = 1
      for d = axis + 1; d < r; d = d + 1 {
        inner = inner * full.shape[d]
      }
      let len = full.shape[axis]
      for o = 0; o < outer; o = o + 1 {
        for i = 0; i < count; i = i + 1 {
          for j = 0; j < inner; j = j + 1 {
            let src = o * count * inner + i * inner + j
            let dst = o * len * inner + (start + i) * inner + j
            full.data[dst] = part.data[src]
          }
        }
      }
      adj[child] = t_add_or(adj[child], full)
    }
  }
}

///|
/// Elementwise addition that keeps the broadcast shape when the
/// shapes differ.
pub fn t_add_or(a : Tensor, b : Tensor) -> Tensor {
  if a.shape.length() == b.shape.length() {
    let mut same = true
    for i = 0; i < a.shape.length(); i = i + 1 {
      if a.shape[i] != b.shape[i] {
        same = false
      }
    }
    if same {
      return match a.add(b) {
        Some(t) => t
        None => a
      }
    }
  }
  match t_add(a, b) {
    Some(t) => t
    None => a
  }
}

///|
/// Elementwise product that keeps the broadcast shape when the
/// shapes differ.
pub fn t_mul_or(a : Tensor, b : Tensor) -> Tensor {
  if a.shape.length() == b.shape.length() {
    let mut same = true
    for i = 0; i < a.shape.length(); i = i + 1 {
      if a.shape[i] != b.shape[i] {
        same = false
      }
    }
    if same {
      return match a.mul(b) {
        Some(t) => t
        None => a
      }
    }
  }
  match t_mul(a, b) {
    Some(t) => t
    None => a
  }
}

///|
/// `d/dx op(x)` elementwise, with the same shape as the operand.
pub fn t_unary_derivative(op : UnaryOp, x : Tensor) -> Tensor {
  let out = Tensor::zeros(x.shape)
  for i = 0; i < x.data.length(); i = i + 1 {
    out.data[i] = unary_derivative(op, x.data[i])
  }
  out
}

///|
/// `d/da (a op b)`, broadcast to `common`.
pub fn t_binary_dleft(
  op : BinaryOp,
  a : Tensor,
  b : Tensor,
  common : Array[Int],
) -> Tensor {
  let out = Tensor::zeros(common)
  let av = match a.broadcast_to(common) {
    Some(t) => t
    None => return out
  }
  let bv = match b.broadcast_to(common) {
    Some(t) => t
    None => return out
  }
  for i = 0; i < out.data.length(); i = i + 1 {
    out.data[i] = binary_dleft(op, av.data[i], bv.data[i])
  }
  out
}

///|
/// `d/db (a op b)`, broadcast to `common`.
pub fn t_binary_dright(
  op : BinaryOp,
  a : Tensor,
  b : Tensor,
  common : Array[Int],
) -> Tensor {
  let out = Tensor::zeros(common)
  let av = match a.broadcast_to(common) {
    Some(t) => t
    None => return out
  }
  let bv = match b.broadcast_to(common) {
    Some(t) => t
    None => return out
  }
  for i = 0; i < out.data.length(); i = i + 1 {
    out.data[i] = binary_dright(op, av.data[i], bv.data[i])
  }
  out
}

///|
/// Re-insert the reduced axis into `gbar` and broadcast it up to
/// the source's shape. This is the backward rule for a reduction:
///
///     out = sum_axis(x, k)   =>   d out / d x[i0, j, i1] = gbar[i0, i1]
///
/// The reduced gradient is missing the summed axis, so every
/// element along that axis maps to the same entry of the gradient.
pub fn t_expand_axis(gbar : Tensor, axis : Int, target : Array[Int]) -> Tensor {
  let out = Tensor::zeros(target)
  let r = target.length()
  let total = out.data.length()
  let coords : Array[Int] = []
  for d = 0; d < r; d = d + 1 {
    coords.push(0)
  }
  let gr = gbar.shape.length()
  for flat = 0; flat < total; flat = flat + 1 {
    // Decode the flat index into per-axis coordinates.
    let mut rest = flat
    let mut d = r - 1
    while d >= 0 {
      coords[d] = rest % target[d]
      rest = rest / target[d]
      d = d - 1
    }
    // Walk the reduced gradient's axes, skipping the one that was
    // summed away. A reduced axis `d` corresponds to source axis
    // `d` before the summed axis and `d + 1` after it.
    let mut offset = 0
    let mut stride = 1
    let mut gd = gr - 1
    while gd >= 0 {
      let cd = if gd < axis { gd } else { gd + 1 }
      offset = offset + coords[cd] * stride
      stride = stride * gbar.shape[gd]
      gd = gd - 1
    }
    out.data[flat] = gbar.data[offset]
  }
  out
}

///|
/// The inverse of an axis permutation.
pub fn t_inverse_perm(perm : Array[Int]) -> Array[Int] {
  let out : Array[Int] = []
  for i = 0; i < perm.length(); i = i + 1 {
    out.push(0)
  }
  for i = 0; i < perm.length(); i = i + 1 {
    out[perm[i]] = i
  }
  out
}
