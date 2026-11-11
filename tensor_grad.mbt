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
#derive(Debug)
pub(all) enum TNode {
  /// A constant, held in the graph's constant pool.
  Const(Int32)
  /// An input tensor, identified by slot.
  Var(Int32)
  /// A read of a whole parameter vector.
  ParamRead(Int32)
  /// An elementwise unary operator.
  Unary(UnaryOp, Int32)
  /// An elementwise binary operator, with broadcasting.
  Binary(BinaryOp, Int32, Int32)
  /// Matrix product.
  MatMul(Int32, Int32)
  /// Sum along one axis, dropping that axis.
  Sum(Int32, Int32)
  /// Mean along one axis, dropping that axis.
  Mean(Int32, Int32)
  /// Mean along one axis, keeping the axis with length one.
  MeanKeep(Int32, Int32)
  /// Reshape to a new shape.
  Reshape(Int32, Array[Int32])
  /// Permute the axes.
  Transpose(Int32, Array[Int32])
  /// Concatenate two tensors along an axis.
  Concat(Int32, Int32, Int32)
  /// A slice along one axis.
  Slice(Int32, Int32, Int32, Int32)
  /// Multiply by a constant tensor held in the pool.
  ScaleBy(Int32, Int32)
  /// Read one row of a parameter vector.
  GatherRow(Int32, Int32, Int32)
  /// `base ^ exponent`, with the exponent carried inline.
  PowBy(Double, Int32)
  /// A 2-D convolution. The operands are the input and the weight;
  /// the geometry lives in the graph's geometry pool.
  Conv2d(Int32, Int32, Int32)
  /// Broadcast a tensor up to a larger shape.
  BroadcastTo(Int32, Array[Int32])
  /// Move a tensor to another shape: broadcast it up when the target
  /// has more axes, sum it down when it has fewer.
  Adjust(Int32, Array[Int32])
  /// Broadcast a tensor up to a shape that has one extra axis, the
  /// shape a reduction came from.
  ExpandAxis(Int32, Int32, Array[Int32])
  /// Pad a tensor with zeros along one axis.
  PadTo(Int32, Array[Int32], Int32, Int32)
  /// Take a window out of a tensor, the inverse of `PadTo`.
  CropTo(Int32, Array[Int32], Int32, Int32)
  /// Place a row of values into a larger tensor, the gradient of a
  /// `GatherRow`.
  ScatterRow(Int32, Array[Int32], Int32, Int32)
  /// One element of a tensor, taken by flat index.
  Select(Int32, Int32)
  /// One element of a scalar put back into a tensor of a given shape.
  Scatter(Int32, Array[Int32], Int32)
  /// The derivative of ReLU: 1 where the input is positive, 0
  /// elsewhere. It is a node rather than a literal precisely so
  /// that a compiled gradient stays correct at a different input.
  ReluDeriv(Int32)
  /// Expand a tensor into a matrix of overlapping patches.
  Im2Col(Int32, Int32)
  /// Scatter a matrix of patches back into a tensor.
  Col2Im(Int32, Int32)
}

///|
pub extend TNode with Debug::{to_repr}

///|
/// A tensor computation under construction.
///
/// Unlike the scalar engine this one does not hash-cons: a tensor
/// node stands for a whole array, so sharing it would save very
/// little, and the index map it would need makes the node
/// unhashable. The scalar engine keeps its own graph for the cases
/// where the sharing actually pays.
#derive(Debug)
pub(all) struct TGraph {
  mut nodes : Array[TNode]
  mut pool : Array[Tensor]
  mut values : Array[Tensor]
  /// The shape of every node's value, filled by the forward pass.
  mut shapes : Array[Array[Int32]]
  /// Set when an operation could not be evaluated. A silent
  /// degradation to a zero scalar would turn a shape mistake into a
  /// wrong gradient, which is much harder to find than an error.
  mut failed : Bool
  /// How many parameter slots have been handed out.
  mut param_count : Int32
  /// Geometry for the operators that need it. An entry is a list of
  /// integers whose meaning depends on the operator:
  ///
  ///   convolution  [in_ch, out_ch, kh, kw, sh, sw, ph, pw]
  ///   im2col       [out_h, out_w, in_ch, kh, kw, sh, sw]
  ///   col2im       [batch, height, width, out_h, out_w, in_ch, kh, kw, sh, sw]
  mut geo : Array[Array[Int32]]
}

///|
pub extend TGraph with Debug::{to_repr}

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
    geo: [],
  }
}

///|
/// The number of nodes.
pub fn TGraph::len(self : TGraph) -> Int32 {
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
pub fn TGraph::constant(self : TGraph, t : Tensor) -> Int32 {
  let slot = self.pool.length()
  self.pool.push(t)
  self.nodes.push(TNode::Const(slot))
  self.nodes.length() - 1
}

///|
/// An input tensor slot, by name of the slot.
pub fn TGraph::variable(self : TGraph, slot : Int32) -> Int32 {
  self.input(slot)
}

///|
/// An input tensor slot.
pub fn TGraph::input(self : TGraph, slot : Int32) -> Int32 {
  self.nodes.push(TNode::Var(slot))
  self.nodes.length() - 1
}

///|
/// A parameter vector slot.
pub fn TGraph::param(self : TGraph, slot : Int32) -> Int32 {
  self.nodes.push(TNode::ParamRead(slot))
  self.nodes.length() - 1
}

///|
/// Allocate a fresh parameter slot and return its index.
///
/// Slots are allocated, never sized. A layer that wants a parameter
/// of a hundred elements still gets one slot; the number of
/// elements is the size of the tensor that fills it.
pub fn TGraph::new_param_slot(self : TGraph) -> Int32 {
  let slot = self.param_count
  self.param_count = self.param_count + 1
  slot
}

///|
/// An elementwise unary operator.
pub fn TGraph::unary(self : TGraph, op : UnaryOp, child : Int32) -> Int32 {
  self.nodes.push(TNode::Unary(op, child))
  self.nodes.length() - 1
}

///|
/// An elementwise binary operator, with broadcasting.
pub fn TGraph::binary(
  self : TGraph,
  op : BinaryOp,
  a : Int32,
  b : Int32,
) -> Int32 {
  self.nodes.push(TNode::Binary(op, a, b))
  self.nodes.length() - 1
}

///|
/// Matrix product.
pub fn TGraph::matmul(self : TGraph, a : Int32, b : Int32) -> Int32 {
  self.nodes.push(TNode::MatMul(a, b))
  self.nodes.length() - 1
}

///|
/// Sum along an axis, dropping that axis.

///|
/// Mean along one axis, dropping the axis.
pub fn TGraph::mean_axis(self : TGraph, child : Int32, axis : Int32) -> Int32 {
  self.nodes.push(TNode::Mean(child, axis))
  self.nodes.length() - 1
}

///|
/// Mean along one axis, keeping the axis with length one.
///
/// Keeping the axis is what a normalisation layer wants: the mean
/// stays broadcastable against the tensor it came from, without the
/// caller having to know the tensor's shape to build the graph.
pub fn TGraph::mean_keep_axis(
  self : TGraph,
  child : Int32,
  axis : Int32,
) -> Int32 {
  self.nodes.push(TNode::MeanKeep(child, axis))
  self.nodes.length() - 1
}

///|
/// Sum along one axis, dropping the axis.
pub fn TGraph::sum_axis(self : TGraph, child : Int32, axis : Int32) -> Int32 {
  self.nodes.push(TNode::Sum(child, axis))
  self.nodes.length() - 1
}

///|
/// Reshape to a new shape.
pub fn TGraph::reshape(
  self : TGraph,
  child : Int32,
  shape : Array[Int32],
) -> Int32 {
  self.nodes.push(TNode::Reshape(child, shape.copy()))
  self.nodes.length() - 1
}

///|
/// Permute the axes.
pub fn TGraph::transpose(
  self : TGraph,
  child : Int32,
  perm : Array[Int32],
) -> Int32 {
  self.nodes.push(TNode::Transpose(child, perm.copy()))
  self.nodes.length() - 1
}

///|
/// Concatenate along an axis.
pub fn TGraph::concat(
  self : TGraph,
  a : Int32,
  b : Int32,
  axis : Int32,
) -> Int32 {
  self.nodes.push(TNode::Concat(a, b, axis))
  self.nodes.length() - 1
}

///|
/// A slice along an axis.
pub fn TGraph::slice(
  self : TGraph,
  child : Int32,
  axis : Int32,
  start : Int32,
  count : Int32,
) -> Int32 {
  self.nodes.push(TNode::Slice(child, axis, start, count))
  self.nodes.length() - 1
}

///|
/// Multiply a node by a constant tensor.
pub fn TGraph::scale_tensor(self : TGraph, x : Int32, k : Tensor) -> Int32 {
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
  param_slot : Int32,
  index : Int32,
  dim : Int32,
) -> Int32 {
  self.nodes.push(TNode::GatherRow(param_slot, index, dim))
  self.nodes.length() - 1
}

///|
/// The shape of a node's value, as recorded by the last forward
/// pass. Returns an empty shape for a node that has not been
/// evaluated.
pub fn TGraph::last_shape(self : TGraph, x : Int32) -> Array[Int32] {
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
pub fn TGraph::pow(self : TGraph, x : Int32, exponent : Double) -> Int32 {
  self.nodes.push(TNode::PowBy(exponent, x))
  self.nodes.length() - 1
}

///|
pub fn TGraph::add(self : TGraph, a : Int32, b : Int32) -> Int32 {
  self.binary(BinaryOp::Add, a, b)
}

///|
pub fn TGraph::sub(self : TGraph, a : Int32, b : Int32) -> Int32 {
  self.binary(BinaryOp::Sub, a, b)
}

///|
pub fn TGraph::mul(self : TGraph, a : Int32, b : Int32) -> Int32 {
  self.binary(BinaryOp::Mul, a, b)
}

///|
pub fn TGraph::div(self : TGraph, a : Int32, b : Int32) -> Int32 {
  self.binary(BinaryOp::Div, a, b)
}

///|
pub fn TGraph::neg(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Neg, x)
}

///|
pub fn TGraph::exp(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Exp, x)
}

///|
pub fn TGraph::ln(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Ln, x)
}

///|
pub fn TGraph::sqrt(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Sqrt, x)
}

///|
pub fn TGraph::sin(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Sin, x)
}

///|
pub fn TGraph::cos(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Cos, x)
}

///|
pub fn TGraph::tanh(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Tanh, x)
}

///|
pub fn TGraph::sigmoid(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Sigmoid, x)
}

///|
pub fn TGraph::softplus(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Softplus, x)
}

///|
pub fn TGraph::relu(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Relu, x)
}

///|
/// The sigmoid linear unit, `x * sigmoid(x)`. Also called swish.
pub fn TGraph::silu(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Silu, x)
}

///|
/// The Gaussian error linear unit, in its tanh approximation.
pub fn TGraph::gelu(self : TGraph, x : Int32) -> Int32 {
  self.unary(UnaryOp::Gelu, x)
}

///|
/// A 2-D convolution node.
///
/// The geometry is stored in the graph's geometry pool rather than
/// in the node, which keeps the node small and lets the backward
/// rules read it without unpacking ten integers every time.
pub fn TGraph::conv2d(
  self : TGraph,
  input : Int32,
  weight : Int32,
  in_channels : Int32,
  out_channels : Int32,
  kernel_h : Int32,
  kernel_w : Int32,
  stride_h : Int32,
  stride_w : Int32,
  pad_h : Int32,
  pad_w : Int32,
) -> Int32 {
  let geo = self.add_geometry([
    in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
  ])
  self.nodes.push(TNode::Conv2d(input, weight, geo))
  self.nodes.length() - 1
}

///|
/// Broadcast a node's value up to `shape`.
pub fn TGraph::broadcast_to(
  self : TGraph,
  x : Int32,
  shape : Array[Int32],
) -> Int32 {
  self.nodes.push(TNode::BroadcastTo(x, shape.copy()))
  self.nodes.length() - 1
}

///|
/// Move a node's value to `shape`.
///
/// The direction follows the rank: a target with more axes means the
/// value was broadcast up and has to be summed down, a target with
/// fewer axes means it was summed away and has to be broadcast back.
/// One node covers both because the two never mix in practice, and
/// because the backward pass needs exactly the same adjustment in
/// the opposite direction.
pub fn TGraph::adjust(self : TGraph, x : Int32, shape : Array[Int32]) -> Int32 {
  self.nodes.push(TNode::Adjust(x, shape.copy()))
  self.nodes.length() - 1
}

///|
/// Broadcast a node's value up to a shape that has one more axis.
pub fn TGraph::expand_axis(
  self : TGraph,
  x : Int32,
  axis : Int32,
  shape : Array[Int32],
) -> Int32 {
  self.nodes.push(TNode::ExpandAxis(x, axis, shape.copy()))
  self.nodes.length() - 1
}

///|
/// Take a window out of a node's value.
pub fn TGraph::crop_to(
  self : TGraph,
  x : Int32,
  shape : Array[Int32],
  axis : Int32,
  start : Int32,
) -> Int32 {
  self.nodes.push(TNode::CropTo(x, shape.copy(), axis, start))
  self.nodes.length() - 1
}

///|
/// One element of a node's value, by flat index.
pub fn TGraph::select(self : TGraph, x : Int32, index : Int32) -> Int32 {
  self.nodes.push(TNode::Select(x, index))
  self.nodes.length() - 1
}

///|
/// Put a node's scalar value back into a tensor of a given shape.
pub fn TGraph::scatter(
  self : TGraph,
  x : Int32,
  shape : Array[Int32],
  index : Int32,
) -> Int32 {
  self.nodes.push(TNode::Scatter(x, shape.copy(), index))
  self.nodes.length() - 1
}

///|
/// Place a node's value into one row of a larger tensor.
pub fn TGraph::scatter_row(
  self : TGraph,
  x : Int32,
  shape : Array[Int32],
  index : Int32,
  dim : Int32,
) -> Int32 {
  self.nodes.push(TNode::ScatterRow(x, shape.copy(), index, dim))
  self.nodes.length() - 1
}

///|
/// A node that reads a constant already in the pool.
pub fn TGraph::pool_const(self : TGraph, slot : Int32) -> Int32 {
  self.nodes.push(TNode::Const(slot))
  self.nodes.length() - 1
}

///|
/// Pad a node's value with zeros along one axis.
pub fn TGraph::pad_to(
  self : TGraph,
  x : Int32,
  shape : Array[Int32],
  axis : Int32,
  start : Int32,
) -> Int32 {
  self.nodes.push(TNode::PadTo(x, shape.copy(), axis, start))
  self.nodes.length() - 1
}

///|
/// The derivative of ReLU, evaluated at run time rather than baked
/// in at compile time.
pub fn TGraph::relu_deriv(self : TGraph, x : Int32) -> Int32 {
  self.nodes.push(TNode::ReluDeriv(x))
  self.nodes.length() - 1
}

///|
/// Store a geometry entry and return its index.
pub fn TGraph::add_geometry(self : TGraph, g : Array[Int32]) -> Int32 {
  self.geo.push(g)
  self.geo.length() - 1
}

///|
/// Expand a node's value into a matrix of overlapping patches.
pub fn TGraph::im2col(self : TGraph, x : Int32, geo : Int32) -> Int32 {
  self.nodes.push(TNode::Im2Col(x, geo))
  self.nodes.length() - 1
}

///|
/// Scatter a matrix of patches back into a tensor.
pub fn TGraph::col2im(self : TGraph, x : Int32, geo : Int32) -> Int32 {
  self.nodes.push(TNode::Col2Im(x, geo))
  self.nodes.length() - 1
}

///|
/// The inputs and parameters a tensor graph is evaluated against.
#derive(Debug)
pub(all) struct TBinding {
  /// Value of each input tensor, indexed by slot.
  inputs : Array[Tensor]
  /// Value of each parameter vector, indexed by slot.
  params : Array[Tensor]
}

///|
pub extend TBinding with Debug::{to_repr}

///|
/// An empty binding.
pub fn TBinding::new() -> TBinding {
  { inputs: [], params: [], }
}

///|
/// Set the value of an input tensor.
pub fn TBinding::set_input(
  self : TBinding,
  slot : Int32,
  value : Tensor,
) -> Unit {
  while self.inputs.length() <= slot {
    self.inputs.push(Tensor::zeros([]))
  }
  self.inputs[slot] = value
}

///|
/// Set the value of a parameter vector.
pub fn TBinding::set_param(
  self : TBinding,
  slot : Int32,
  value : Tensor,
) -> Unit {
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
  id : Int32,
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
    TNode::Mean(child, axis) => {
      let x = values[child]
      match t_sum_axis(x, axis) {
        Some(total) => {
          let n = x.shape[axis]
          if n <= 0 {
            g.failed = true
            return Tensor::zeros([])
          }
          t_scale(total, 1.0 / n.to_double())
        }
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    }
    TNode::MeanKeep(child, axis) => {
      let x = values[child]
      match t_sum_axis_keep(x, axis) {
        Some(total) => {
          let n = x.shape[axis]
          if n <= 0 {
            g.failed = true
            return Tensor::zeros([])
          }
          t_scale(total, 1.0 / n.to_double())
        }
        None => {
          g.failed = true
          Tensor::zeros([])
        }
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
    TNode::BroadcastTo(child, shape) =>
      match values[child].broadcast_to(shape) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::Adjust(child, shape) => t_sum_to_shape(values[child], shape)
    TNode::ExpandAxis(child, axis, shape) =>
      t_expand_axis(values[child], axis, shape)
    TNode::CropTo(child, shape, axis, start) =>
      match t_crop_to(values[child], shape, axis, start) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::ScatterRow(child, shape, index, dim) =>
      match t_scatter_row(values[child], shape, index, dim) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::Select(child, index) => {
      let x = values[child]
      if index < 0 || index >= x.data.length() {
        g.failed = true
        return Tensor::zeros([])
      }
      Tensor::filled([], x.data[index])
    }
    TNode::Scatter(child, shape, index) => {
      let x = values[child]
      if x.data.length() != 1 {
        g.failed = true
        return Tensor::zeros([])
      }
      let out = Tensor::zeros(shape)
      if index >= 0 && index < out.data.length() {
        out.data[index] = x.data[0]
      } else {
        g.failed = true
      }
      out
    }
    TNode::PadTo(child, shape, axis, start) =>
      match t_pad_to(values[child], shape, axis, start) {
        Some(t) => t
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
    TNode::ReluDeriv(child) => {
      // The whole point of this node: the derivative is decided at
      // evaluation time, so a compiled gradient stays correct when
      // it is re-evaluated at a different input.
      let x = values[child]
      let out = Tensor::zeros(x.shape)
      for i = 0; i < x.data.length(); i = i + 1 {
        out.data[i] = if x.data[i] > 0.0 { 1.0 } else { 0.0 }
      }
      out
    }
    TNode::Im2Col(child, geo) => {
      let gv = g.geo[geo]
      t_im2col(values[child], gv[0], gv[1], gv[2], gv[3], gv[4], gv[5], gv[6])
    }
    TNode::Col2Im(child, geo) => {
      let gv = g.geo[geo]
      t_col2im(
        values[child],
        gv[0],
        gv[1],
        gv[2],
        gv[3],
        gv[4],
        gv[5],
        gv[6],
        gv[7],
        gv[8],
        gv[9],
      )
    }
    TNode::Conv2d(input, weight, geo) => {
      let gv = g.geo[geo]
      let stride_h = gv[4]
      let stride_w = gv[5]
      let pad_h = gv[6]
      let pad_w = gv[7]
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
        None => {
          g.failed = true
          Tensor::zeros([])
        }
      }
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

///|
/// A compiled tensor gradient.
///
/// The gradient is a *program*, not a number. Every adjoint is a node
/// in `graph`, sitting above the primal's nodes, and evaluating the
/// graph evaluates the gradient along with it. That is what lets a
/// compiled gradient be reused at a different input, and what makes
/// a second derivative a matter of differentiating the gradient
/// rather than writing a second engine.
#derive(Debug)
pub(all) struct TGradGraph {
  /// The graph the sweep emitted into. Its first `primal_size` nodes
  /// mirror the primal's; everything above is gradient arithmetic.
  graph : TGraph
  /// How many leading nodes came from the primal.
  primal_size : Int32
  /// Node in `graph` holding `d(target) / d(input_i)`, one per input
  /// slot.
  input_adj : Array[Int32]
  /// Node in `graph` holding `d(target) / d(param_i)`, one per
  /// parameter slot.
  param_adj : Array[Int32]
  /// The shape each input had when the gradient was compiled.
  ///
  /// The backward rules bake the operand shapes into the program,
  /// so the program is valid for one set of shapes and not another.
  /// Recording them lets the evaluation check, rather than quietly
  /// returning the gradient of a differently-shaped function.
  input_shapes : Array[Array[Int32]]
  /// Value of the differentiated node under the binding the gradient
  /// was compiled against.
  output : Tensor
}

///|
pub extend TGradGraph with Debug::{to_repr}

///|
/// A tensor graph paired with its compiled gradient.
#derive(Debug)
pub(all) struct TProgram {
  primal : TGraph
  grad : TGradGraph
}

///|
pub extend TProgram with Debug::{to_repr}

///|
/// The result of a backward pass.
#derive(Debug)
pub(all) struct TGrad {
  /// Gradient with respect to each input tensor.
  input_grads : Array[Tensor]
  /// Gradient with respect to each parameter vector.
  param_grads : Array[Tensor]
  /// Value of the differentiated node.
  output : Tensor
}

///|
pub extend TGrad with Debug::{to_repr}

///|
/// Build a graph with the same node sequence as `g`.
///
/// Mirroring rather than sharing is deliberate: the backward sweep
/// appends gradient nodes to the mirror while leaving the primal
/// untouched, and a single shared node array would make the primal
/// impossible to keep after differentiation.
pub fn t_mirror(g : TGraph) -> TGraph {
  let h = TGraph::new()
  for i = 0; i < g.pool.length(); i = i + 1 {
    h.pool.push(g.pool[i])
  }
  for i = 0; i < g.geo.length(); i = i + 1 {
    h.geo.push(g.geo[i].copy())
  }
  for i = 0; i < g.nodes.length(); i = i + 1 {
    h.nodes.push(g.nodes[i])
  }
  h
}

///|
/// Compile the gradient of `target` with respect to every input and
/// every parameter.
///
/// The result is a program. Evaluating `grad.graph` against a binding
/// evaluates the gradient at that binding's values, so the same
/// compiled gradient can be re-used at every step of a training run,
/// or differentiated again to get a second derivative.
///
/// The backward rules need the primal's values to know each operand's
/// shape, so the primal is evaluated once during compilation. The
/// shapes it bakes in are the shapes of the graph, which do not
/// change between evaluations.
pub fn compile_tensor_gradient(
  g : TGraph,
  b : TBinding,
  target : Int32,
) -> TGradGraph {
  let values = t_forward(g, b)
  let n = g.nodes.length()
  let h = t_mirror(g)
  // A shared zero, so that a node nobody contributed to can be
  // recognised by identity and skipped.
  let zero = h.constant(Tensor::zeros([]))
  let adj : Array[Int32] = []
  for i = 0; i < n; i = i + 1 {
    adj.push(zero)
  }
  let input_adj : Array[Int32] = []
  for i = 0; i < b.inputs.length(); i = i + 1 {
    input_adj.push(h.constant(Tensor::zeros(b.inputs[i].shape)))
  }
  let param_adj : Array[Int32] = []
  for i = 0; i < b.params.length(); i = i + 1 {
    param_adj.push(h.constant(Tensor::zeros(b.params[i].shape)))
  }
  let input_shapes : Array[Array[Int32]] = []
  for i = 0; i < b.inputs.length(); i = i + 1 {
    input_shapes.push(b.inputs[i].shape.copy())
  }
  if target < 0 || target >= n {
    return {
      graph: h,
      primal_size: n,
      input_adj,
      param_adj,
      input_shapes,
      output: Tensor::zeros([]),
    }
  }
  // The adjoint of the differentiated node is a tensor of ones: every
  // element of the output contributes to the derivative.
  adj[target] = h.constant(Tensor::filled(values[target].shape, 1.0))
  for i = n - 1; i >= 0; i = i - 1 {
    if adj[i] == zero {
      continue
    }
    t_push_adjoints(g, h, b, values, adj, input_adj, param_adj, i)
  }
  {
    graph: h,
    primal_size: n,
    input_adj,
    param_adj,
    input_shapes,
    output: values[target],
  }
}

///|
/// Whether the binding has the shapes the program was compiled for.
///
/// A compiled gradient is valid for the shapes it was compiled
/// against: the backward rules bake the operand shapes into the
/// program, because the shapes are what say how a broadcast has to
/// be summed back down. Evaluating the same program against
/// differently-shaped inputs would therefore compute the gradient of
/// a different function, quietly. This check is what turns that into
/// a reportable condition.
pub fn TGradGraph::shapes_match(self : TGradGraph, b : TBinding) -> Bool {
  if b.inputs.length() < self.input_shapes.length() {
    return false
  }
  for i = 0; i < self.input_shapes.length(); i = i + 1 {
    let want = self.input_shapes[i]
    let got = b.inputs[i].shape
    if want.length() != got.length() {
      return false
    }
    for d = 0; d < want.length(); d = d + 1 {
      if want[d] != got[d] {
        return false
      }
    }
  }
  true
}

///|
/// Evaluate a compiled gradient at a binding.
///
/// The graph carries the primal and the gradient together, so one
/// evaluation produces both. Evaluating the same program at a
/// different binding gives the gradient at that binding, which is
/// the whole reason the gradient is a program and not a number.
pub fn evaluate_tensor_gradient(
  grad : TGradGraph,
  b : TBinding,
) -> (Array[Tensor], Array[Tensor]) {
  let values = t_forward(grad.graph, b)
  let input_grads : Array[Tensor] = []
  for i = 0; i < grad.input_adj.length(); i = i + 1 {
    input_grads.push(values[grad.input_adj[i]])
  }
  let param_grads : Array[Tensor] = []
  for i = 0; i < grad.param_adj.length(); i = i + 1 {
    param_grads.push(values[grad.param_adj[i]])
  }
  (input_grads, param_grads)
}

///|
/// Differentiate `target` with respect to every input and every
/// parameter, and return the numbers.
///
/// This is the convenience form of `compile_tensor_gradient` followed
/// by `evaluate_tensor_gradient`. The program underneath is the same
/// one; this only collapses it to numbers.
pub fn t_grad(g : TGraph, b : TBinding, target : Int32) -> TGrad {
  let grad = compile_tensor_gradient(g, b, target)
  let (input_grads, param_grads) = evaluate_tensor_gradient(grad, b)
  { input_grads, param_grads, output: grad.output, }
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
/// tensor's shape is the broadcast shape. Aligning the two from
/// the right gives the axes to sum:
///
///   * axes the target does not have at all — the operand was
///     broadcast along a whole new axis;
///   * axes where the target has length one and the broadcast shape
///     has more — the operand's single element was repeated.
///
/// The result has exactly the target's shape, which is what the
/// caller needs to add it into the operand's adjoint.
pub fn t_sum_to_shape(t : Tensor, target : Array[Int32]) -> Tensor {
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
pub fn t_sum_axis_keep(t : Tensor, axis : Int32) -> Tensor? {
  if axis < 0 || axis >= t.shape.length() {
    return None
  }
  let out_shape = t.shape.copy()
  out_shape[axis] = 1
  let out = Tensor::zeros(out_shape)
  let mut outer = 1
  for d = 0; d < axis; d = d + 1 {
    outer = outer * t.shape[d]
  }
  let mut inner = 1
  for d = axis + 1; d < t.shape.length(); d = d + 1 {
    inner = inner * t.shape[d]
  }
  let len = t.shape[axis]
  for o = 0; o < outer; o = o + 1 {
    for i = 0; i < len; i = i + 1 {
      for j = 0; j < inner; j = j + 1 {
        // The source is read with the original strides; the
        // destination is written with the reduced ones, because the
        // output's axis has length one and its buffer is a
        // correspondingly smaller.
        let src = o * len * inner + i * inner + j
        let dst = o * inner + j
        out.data[dst] = out.data[dst] + t.data[src]
      }
    }
  }
  Some(out)
}

///|
/// The local derivative of a unary operator, as an expression.
///
/// The expression is the important part. A number would freeze the
/// derivative at the point the gradient was compiled, and a gradient
/// that cannot be re-evaluated is a gradient that is wrong the moment
/// the input moves. ReLU is the sharp case: its derivative is a step
/// function, so it has to be *computed* where the gradient is
/// evaluated rather than remembered.
fn t_derivative_expr(h : TGraph, op : UnaryOp, x : Int32) -> Int32 {
  let one = h.constant(Tensor::filled([], 1.0))
  match op {
    UnaryOp::Neg => h.constant(Tensor::filled([], -1.0))
    UnaryOp::Exp => h.exp(x)
    UnaryOp::Ln => h.div(one, x)
    UnaryOp::Sqrt => {
      let half = h.constant(Tensor::filled([], 0.5))
      h.mul(half, h.div(one, h.sqrt(x)))
    }
    UnaryOp::Sin => h.cos(x)
    UnaryOp::Cos => h.neg(h.sin(x))
    UnaryOp::Tanh => {
      // 1 - t^2 with t = tanh(x)
      let t = h.tanh(x)
      h.sub(one, h.mul(t, t))
    }
    UnaryOp::Sigmoid => {
      // s(1 - s) with s = sigmoid(x)
      let s = h.sigmoid(x)
      h.mul(s, h.sub(one, s))
    }
    UnaryOp::Softplus => h.sigmoid(x)
    UnaryOp::Relu => h.relu_deriv(x)
    UnaryOp::Silu => {
      // s + x s (1 - s) with s = sigmoid(x)
      let s = h.sigmoid(x)
      h.add(s, h.mul(x, h.mul(s, h.sub(one, s))))
    }
    UnaryOp::Gelu => {
      // The derivative of 0.5 x (1 + tanh u), with
      // u = c (x + 0.044715 x^3).
      let x3 = h.pow(x, 3.0)
      let x2 = h.pow(x, 2.0)
      let inner = h.add(x, h.mul(h.constant(Tensor::filled([], 0.044715)), x3))
      let u = h.mul(h.constant(Tensor::filled([], gelu_c)), inner)
      let t = h.tanh(u)
      let du = h.mul(
        h.constant(Tensor::filled([], gelu_c)),
        h.add(one, h.mul(h.constant(Tensor::filled([], 3.0 * 0.044715)), x2)),
      )
      let first = h.mul(h.constant(Tensor::filled([], 0.5)), h.add(one, t))
      let second = h.mul(
        h.constant(Tensor::filled([], 0.5)),
        h.mul(x, h.mul(h.sub(one, h.mul(t, t)), du)),
      )
      h.add(first, second)
    }
  }
}

///|
/// Partial derivative of `a op b` with respect to `a`, as an
/// expression.
fn t_derivative_left(h : TGraph, op : BinaryOp, b : Int32) -> Int32 {
  match op {
    BinaryOp::Add => h.constant(Tensor::filled([], 1.0))
    BinaryOp::Sub => h.constant(Tensor::filled([], 1.0))
    BinaryOp::Mul => b
    BinaryOp::Div => h.div(h.constant(Tensor::filled([], 1.0)), b)
  }
}

///|
/// Partial derivative of `a op b` with respect to `b`, as an
/// expression.
fn t_derivative_right(h : TGraph, op : BinaryOp, a : Int32, b : Int32) -> Int32 {
  match op {
    BinaryOp::Add => h.constant(Tensor::filled([], 1.0))
    BinaryOp::Sub => h.constant(Tensor::filled([], -1.0))
    BinaryOp::Mul => a
    BinaryOp::Div => {
      // -a / b^2
      let numerator = h.neg(a)
      let denominator = h.mul(b, b)
      h.div(numerator, denominator)
    }
  }
}

///|
/// Push the adjoint of node `i` onto its operands.
///
/// Every rule emits an *expression* into the gradient graph rather
/// than a number. The expression mentions the primal's own nodes, so
/// evaluating the gradient graph at any binding evaluates the
/// derivative there.
fn t_push_adjoints(
  g : TGraph,
  h : TGraph,
  b : TBinding,
  values : Array[Tensor],
  adj : Array[Int32],
  input_adj : Array[Int32],
  param_adj : Array[Int32],
  i : Int32,
) -> Unit {
  let gbar = adj[i]
  match g.nodes[i] {
    TNode::Const(_) => ()
    TNode::Var(slot) =>
      if slot >= 0 && slot < input_adj.length() {
        input_adj[slot] = h.add(input_adj[slot], gbar)
      }
    TNode::ParamRead(slot) =>
      if slot >= 0 && slot < param_adj.length() {
        param_adj[slot] = h.add(param_adj[slot], gbar)
      }
    TNode::Unary(op, child) => {
      let deriv = t_derivative_expr(h, op, child)
      let contrib = h.mul(gbar, deriv)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Binary(op, a, c) => {
      // The two partial derivatives are evaluated in the broadcast
      // shape and then summed back down to each operand's shape.
      let da = t_derivative_left(h, op, c)
      let db = t_derivative_right(h, op, a, c)
      let contrib_a = h.mul(gbar, da)
      let contrib_b = h.mul(gbar, db)
      adj[a] = h.add(adj[a], h.adjust(contrib_a, values[a].shape))
      adj[c] = h.add(adj[c], h.adjust(contrib_b, values[c].shape))
    }
    TNode::MatMul(a, c) => {
      // d/da = gbar @ c^T, d/dc = a^T @ gbar
      let ct = h.transpose(c, [1, 0])
      let at = h.transpose(a, [1, 0])
      adj[a] = h.add(adj[a], h.matmul(gbar, ct))
      adj[c] = h.add(adj[c], h.matmul(at, gbar))
    }
    TNode::Sum(child, axis) => {
      // d(sum_axis(x))/dx is the gradient broadcast back up to x's
      // shape, with the summed axis re-inserted as a size-1 axis.
      let contrib = h.expand_axis(gbar, axis, values[child].shape)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Mean(child, axis) => {
      // A mean is a sum divided by the axis's length, so the
      // gradient is the broadcast-back gradient scaled by the same
      // factor.
      let n = values[child].shape[axis]
      if n <= 0 {
        return
      }
      let expanded = h.expand_axis(gbar, axis, values[child].shape)
      let contrib = h.mul(
        expanded,
        h.constant(Tensor::filled([], 1.0 / n.to_double())),
      )
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::MeanKeep(child, axis) => {
      // The axis was kept, so the gradient is broadcast back along
      // it and scaled by the axis's length.
      let n = values[child].shape[axis]
      if n <= 0 {
        return
      }
      let expanded = h.broadcast_to(gbar, values[child].shape)
      let contrib = h.mul(
        expanded,
        h.constant(Tensor::filled([], 1.0 / n.to_double())),
      )
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Reshape(child, _) => {
      let contrib = h.reshape(gbar, values[child].shape)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Transpose(child, perm) => {
      let inverse = t_inverse_perm(perm)
      let contrib = h.transpose(gbar, inverse)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Concat(a, c, axis) => {
      let la = values[a].shape[axis]
      let lc = values[c].shape[axis]
      let pa = h.slice(gbar, axis, 0, la)
      let pc = h.slice(gbar, axis, la, lc)
      adj[a] = h.add(adj[a], pa)
      adj[c] = h.add(adj[c], pc)
    }
    TNode::Slice(child, axis, start, _count) => {
      // The gradient of a slice is the incoming gradient put back
      // where the slice came from, with zeros everywhere else.
      let contrib = h.pad_to(gbar, values[child].shape, axis, start)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::ScaleBy(child, slot) => {
      // d/dx = gbar * k, with k the same constant.
      let contrib = h.mul(gbar, h.pool_const(slot))
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::PowBy(exponent, child) => {
      // d/dx x^c = c * x^(c-1)
      let c = h.constant(Tensor::filled([], exponent))
      let d = h.mul(c, h.pow(child, exponent - 1.0))
      let contrib = h.mul(gbar, d)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::GatherRow(slot, index, dim) =>
      // The gradient of a row read goes back into that row of the
      // parameter it came from.
      if slot >= 0 && slot < param_adj.length() {
        let shape = b.params[slot].shape
        let contrib = h.scatter_row(gbar, shape, index, dim)
        param_adj[slot] = h.add(param_adj[slot], contrib)
      }
    TNode::Conv2d(input, weight, geo) =>
      t_push_conv_adjoints(g, h, values, adj, gbar, input, weight, geo)
    TNode::BroadcastTo(child, _shape) => {
      // The inverse of a broadcast is a sum back down.
      let contrib = h.adjust(gbar, values[child].shape)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Adjust(child, _shape) => {
      // The inverse of a sum back down is a broadcast up.
      let contrib = h.broadcast_to(gbar, values[child].shape)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::ExpandAxis(child, axis, _shape) => {
      // The value was broadcast along the axis, so the gradient sums
      // over it.
      let contrib = h.sum_axis(gbar, axis)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::PadTo(child, _shape, axis, start) => {
      let contrib = h.crop_to(gbar, values[child].shape, axis, start)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::CropTo(child, _shape, axis, start) => {
      let contrib = h.pad_to(gbar, values[child].shape, axis, start)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::ReluDeriv(child) =>
      // ReLU's derivative is a step function, so its own derivative is
      // zero almost everywhere. The point where it is not is the
      // kink, where the derivative does not exist.
      ignore(child)
    TNode::Im2Col(child, geo) => {
      // The gradient of a patch expansion is a patch accumulation.
      let gv = g.geo[geo]
      let child_shape = values[child].shape
      if child_shape.length() != 4 {
        return
      }
      let col_geo = h.add_geometry([
        child_shape[0],
        child_shape[2],
        child_shape[3],
        gv[0],
        gv[1],
        gv[2],
        gv[3],
        gv[4],
        gv[5],
        gv[6],
      ])
      let contrib = h.col2im(gbar, col_geo)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Col2Im(child, geo) => {
      // The gradient of a patch accumulation is a patch expansion.
      let gv = g.geo[geo]
      let patch_geo = h.add_geometry([
        gv[3],
        gv[4],
        gv[5],
        gv[6],
        gv[7],
        gv[8],
        gv[9],
      ])
      let contrib = h.im2col(gbar, patch_geo)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::ScatterRow(child, _shape, index, dim) => {
      // The gradient picks out the row that was written.
      let row = h.slice(gbar, 0, index, 1)
      let contrib = h.reshape(row, [dim])
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Select(child, index) => {
      // The gradient is the selected element put back where it came
      // from.
      let contrib = h.scatter(gbar, values[child].shape, index)
      adj[child] = h.add(adj[child], contrib)
    }
    TNode::Scatter(child, _shape, index) => {
      // The gradient is the element that was written.
      let contrib = h.select(gbar, index)
      adj[child] = h.add(adj[child], contrib)
    }
  }
}

///|
/// Push the adjoint of a convolution onto its input and its weight.
///
/// The convolution is differentiated through its matrix form: the
/// input is expanded into a matrix of overlapping patches, and the
/// two gradients are ordinary matrix products. Writing it this way
/// keeps the rule short and, more importantly, keeps it an
/// *expression* — the patches are a node, not a number, so the
/// convolution's gradient is reusable like every other.
fn t_push_conv_adjoints(
  g : TGraph,
  h : TGraph,
  values : Array[Tensor],
  adj : Array[Int32],
  gbar : Int32,
  input : Int32,
  weight : Int32,
  geo : Int32,
) -> Unit {
  let gv = g.geo[geo]
  let in_channels = gv[0]
  let out_channels = gv[1]
  let kernel_h = gv[2]
  let kernel_w = gv[3]
  let stride_h = gv[4]
  let stride_w = gv[5]
  let pad_h = gv[6]
  let pad_w = gv[7]
  let xi = values[input]
  if xi.shape.length() != 4 {
    return
  }
  let batch = xi.shape[0]
  let channels = xi.shape[1]
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
  let gbar_t = h.transpose(gbar, [0, 2, 3, 1])
  let gbar_rows = h.reshape(gbar_t, [batch * out_h * out_w, out_channels])
  // The weight flattened to (out_ch, row_len).
  let w_flat = h.reshape(weight, [out_channels, row_len])
  // d/d(patches) = d/out @ w
  let d_patches = h.matmul(gbar_rows, w_flat)
  // The patches themselves, from the padded input.
  let patch_geo = h.add_geometry([
    out_h, out_w, in_channels, kernel_h, kernel_w, stride_h, stride_w,
  ])
  let padded_h = h.pad_to(
    input,
    [batch, channels, height + 2 * pad_h, width],
    2,
    pad_h,
  )
  let padded = h.pad_to(
    padded_h,
    [batch, channels, height + 2 * pad_h, width + 2 * pad_w],
    3,
    pad_w,
  )
  let patches = h.im2col(padded, patch_geo)
  // d/d(weight) = patches^T @ d/out
  let patches_t = h.transpose(patches, [1, 0])
  let d_w_rows = h.matmul(patches_t, gbar_rows)
  let d_w_t = h.transpose(d_w_rows, [1, 0])
  let d_w = h.reshape(d_w_t, [out_channels, in_channels, kernel_h, kernel_w])
  // d/d(input) = col2im(d/d(patches)), cropped back out of the
  // padding.
  let col_geo = h.add_geometry([
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
  ])
  let d_padded = h.col2im(d_patches, col_geo)
  let d_input_h = h.crop_to(
    d_padded,
    [batch, channels, height, width + 2 * pad_w],
    2,
    pad_h,
  )
  let d_input = h.crop_to(d_input_h, [batch, channels, height, width], 3, pad_w)
  adj[input] = h.add(adj[input], d_input)
  adj[weight] = h.add(adj[weight], d_w)
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
  common : Array[Int32],
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
  common : Array[Int32],
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
pub fn t_expand_axis(
  gbar : Tensor,
  axis : Int32,
  target : Array[Int32],
) -> Tensor {
  let out = Tensor::zeros(target)
  let r = target.length()
  let total = out.data.length()
  let coords : Array[Int32] = []
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
pub fn t_inverse_perm(perm : Array[Int32]) -> Array[Int32] {
  let out : Array[Int32] = []
  for i = 0; i < perm.length(); i = i + 1 {
    out.push(0)
  }
  for i = 0; i < perm.length(); i = i + 1 {
    out[perm[i]] = i
  }
  out
}
