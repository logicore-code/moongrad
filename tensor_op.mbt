// tensor_op.mbt
//
// Tensor operations.
//
// Everything here is a *host-side* operation on concrete tensor
// values. The differentiable versions live in `tensor_grad.mbt`, and
// they are written in terms of these.
//
// The broadcasting rules are numpy's: two shapes broadcast when,
// aligned from the right, every axis is either equal or one of them
// is 1. `broadcast_map` turns that into an index table, and every
// elementwise operator is then the same loop with a different
// combination of the two operands.

///|
/// Element-wise sum, broadcasting the operands first.
pub fn t_add(a : Tensor, b : Tensor) -> Tensor? {
  let shape = match common_shape(a.shape, b.shape) {
    Some(s) => s
    None => return None
  }
  let x = match a.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let y = match b.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  x.add(y)
}

///|
/// Element-wise difference, broadcasting the operands first.
pub fn t_sub(a : Tensor, b : Tensor) -> Tensor? {
  let shape = match common_shape(a.shape, b.shape) {
    Some(s) => s
    None => return None
  }
  let x = match a.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let y = match b.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  x.sub(y)
}

///|
/// Element-wise product, broadcasting the operands first.
pub fn t_mul(a : Tensor, b : Tensor) -> Tensor? {
  let shape = match common_shape(a.shape, b.shape) {
    Some(s) => s
    None => return None
  }
  let x = match a.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let y = match b.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  x.mul(y)
}

///|
/// Element-wise quotient, broadcasting the operands first.
pub fn t_div(a : Tensor, b : Tensor) -> Tensor? {
  let shape = match common_shape(a.shape, b.shape) {
    Some(s) => s
    None => return None
  }
  let x = match a.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let y = match b.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let out = Tensor::zeros(shape)
  for i = 0; i < x.data.length(); i = i + 1 {
    out.data[i] = x.data[i] / y.data[i]
  }
  Some(out)
}

///|
/// Negate every element.
pub fn t_neg(a : Tensor) -> Tensor {
  a.neg()
}

///|
/// Multiply every element by a scalar.
pub fn t_scale(a : Tensor, k : Double) -> Tensor {
  a.scale(k)
}

///|
/// Add a scalar to every element.
pub fn t_shift(a : Tensor, k : Double) -> Tensor {
  a.shift(k)
}

///|
/// `exp` of every element.
pub fn t_exp(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = @math.exp(a.data[i])
  }
  out
}

///|
/// Natural logarithm of every element. Elements that are not
/// positive produce a negative infinity rather than a silent NaN,
/// so a caller can see the problem.
pub fn t_log(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    let x = a.data[i]
    out.data[i] = if x > 0.0 { @math.ln(x) } else { -1.0 / 0.0 }
  }
  out
}

///|
/// Square root of every element.
pub fn t_sqrt(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    let x = a.data[i]
    out.data[i] = if x >= 0.0 { x.sqrt() } else { -1.0 / 0.0 }
  }
  out
}

///|
/// Sine of every element.
pub fn t_sin(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = @math.sin(a.data[i])
  }
  out
}

///|
/// Cosine of every element.
pub fn t_cos(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = @math.cos(a.data[i])
  }
  out
}

///|
/// Hyperbolic tangent of every element.
pub fn t_tanh(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = @math.tanh(a.data[i])
  }
  out
}

///|
/// Logistic sigmoid of every element.
pub fn t_sigmoid(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = 1.0 / (1.0 + @math.exp(-a.data[i]))
  }
  out
}

///|
/// ReLU of every element.
pub fn t_relu(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    out.data[i] = if a.data[i] > 0.0 { a.data[i] } else { 0.0 }
  }
  out
}

///|
/// `log(1 + e^x)`, evaluated as `max(x, 0) + log(1 + e^-|x|)` so it
/// does not overflow for large `x`.
pub fn t_softplus(a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    let x = a.data[i]
    let mx = if x > 0.0 { x } else { 0.0 }
    let ax = if x < 0.0 { -x } else { x }
    out.data[i] = mx + @math.ln(1.0 + @math.exp(-ax))
  }
  out
}

///|
/// Matrix product. `(m, k) * (k, n) -> (m, n)`.
pub fn t_matmul(a : Tensor, b : Tensor) -> Tensor? {
  a.matmul(b)
}

///|
/// Transpose the last two axes of a rank-2 tensor.
pub fn t_transpose(a : Tensor) -> Tensor? {
  if a.shape.length() != 2 {
    return None
  }
  a.transpose([1, 0])
}

///|
/// Sum along one axis, dropping that axis from the result.
///
/// The output's shape has one axis fewer than the input's, and the
/// data is laid out so that the surviving axes stay in order.
pub fn t_sum_axis(a : Tensor, axis : Int32) -> Tensor? {
  if axis < 0 || axis >= a.shape.length() {
    return None
  }
  let r = a.shape.length()
  let mut outer = 1
  for d = 0; d < axis; d = d + 1 {
    outer = outer * a.shape[d]
  }
  let mut inner = 1
  for d = axis + 1; d < r; d = d + 1 {
    inner = inner * a.shape[d]
  }
  let len = a.shape[axis]
  let out_shape = a.shape.copy()
  ignore(out_shape.remove(axis))
  let out = Tensor::zeros(out_shape)
  for o = 0; o < outer; o = o + 1 {
    for i = 0; i < len; i = i + 1 {
      for j = 0; j < inner; j = j + 1 {
        let src = o * len * inner + i * inner + j
        let dst = o * inner + j
        out.data[dst] = out.data[dst] + a.data[src]
      }
    }
  }
  Some(out)
}

///|
/// Mean along one axis.
pub fn t_mean_axis(a : Tensor, axis : Int32) -> Tensor? {
  let summed = match t_sum_axis(a, axis) {
    Some(t) => t
    None => return None
  }
  let n = a.shape[axis].to_double()
  let out = Tensor::zeros(summed.shape)
  for i = 0; i < summed.data.length(); i = i + 1 {
    out.data[i] = summed.data[i] / n
  }
  Some(out)
}

///|
/// Sum of every element.
pub fn t_sum(a : Tensor) -> Double {
  a.sum()
}

///|
/// Mean of every element.
pub fn t_mean(a : Tensor) -> Double {
  a.mean()
}

///|
/// Concatenate two tensors along an axis.
pub fn t_concat(a : Tensor, b : Tensor, axis : Int32) -> Tensor? {
  a.concat(b, axis)
}

///|
/// Softmax along the last axis. Computed with the max-subtraction
/// trick so the exponentials cannot overflow.
pub fn t_softmax(a : Tensor) -> Tensor? {
  if a.shape.length() == 0 {
    return None
  }
  let r = a.shape.length()
  let last = a.shape[r - 1]
  let mut outer = 1
  for d = 0; d < r - 1; d = d + 1 {
    outer = outer * a.shape[d]
  }
  let out = Tensor::zeros(a.shape)
  for o = 0; o < outer; o = o + 1 {
    let base = o * last
    let mut m = a.data[base]
    for i = 1; i < last; i = i + 1 {
      if a.data[base + i] > m {
        m = a.data[base + i]
      }
    }
    let mut total = 0.0
    for i = 0; i < last; i = i + 1 {
      let e = @math.exp(a.data[base + i] - m)
      out.data[base + i] = e
      total = total + e
    }
    for i = 0; i < last; i = i + 1 {
      out.data[base + i] = out.data[base + i] / total
    }
  }
  Some(out)
}

///|
/// Log-softmax along the last axis, which is the stable form of
/// `log(softmax(x))` and the one a cross-entropy loss wants.
pub fn t_log_softmax(a : Tensor) -> Tensor? {
  if a.shape.length() == 0 {
    return None
  }
  let r = a.shape.length()
  let last = a.shape[r - 1]
  let mut outer = 1
  for d = 0; d < r - 1; d = d + 1 {
    outer = outer * a.shape[d]
  }
  let out = Tensor::zeros(a.shape)
  for o = 0; o < outer; o = o + 1 {
    let base = o * last
    let mut m = a.data[base]
    for i = 1; i < last; i = i + 1 {
      if a.data[base + i] > m {
        m = a.data[base + i]
      }
    }
    let mut total = 0.0
    for i = 0; i < last; i = i + 1 {
      total = total + @math.exp(a.data[base + i] - m)
    }
    let lse = m + @math.ln(total)
    for i = 0; i < last; i = i + 1 {
      out.data[base + i] = a.data[base + i] - lse
    }
  }
  Some(out)
}

///|
/// Cross-entropy of a batch of logits against integer labels.
///
/// `logits` is `(batch, classes)` and `labels` holds one class index
/// per row. The mean over the batch is returned.
pub fn t_cross_entropy(logits : Tensor, labels : Array[Int32]) -> Double? {
  if logits.shape.length() != 2 {
    return None
  }
  let batch = logits.shape[0]
  let classes = logits.shape[1]
  if labels.length() != batch {
    return None
  }
  let mut total = 0.0
  for r = 0; r < batch; r = r + 1 {
    let label = labels[r]
    if label < 0 || label >= classes {
      return None
    }
    let base = r * classes
    let mut m = logits.data[base]
    for i = 1; i < classes; i = i + 1 {
      if logits.data[base + i] > m {
        m = logits.data[base + i]
      }
    }
    let mut sum = 0.0
    for i = 0; i < classes; i = i + 1 {
      sum = sum + @math.exp(logits.data[base + i] - m)
    }
    let lse = m + @math.ln(sum)
    total = total + (lse - logits.data[base + label])
  }
  Some(total / batch.to_double())
}

///|
/// Mean squared error between two tensors of the same shape.
pub fn t_mse(pred : Tensor, target : Tensor) -> Double? {
  if pred.shape.length() != target.shape.length() {
    return None
  }
  for i = 0; i < pred.shape.length(); i = i + 1 {
    if pred.shape[i] != target.shape[i] {
      return None
    }
  }
  let mut acc = 0.0
  for i = 0; i < pred.data.length(); i = i + 1 {
    let d = pred.data[i] - target.data[i]
    acc = acc + d * d
  }
  if pred.data.length() == 0 {
    return Some(0.0)
  }
  Some(acc / pred.data.length().to_double())
}

///|
/// Apply a unary operator elementwise, preserving the shape.
pub fn t_unary(op : UnaryOp, a : Tensor) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    let x = a.data[i]
    let v = match op {
      UnaryOp::Neg => -x
      UnaryOp::Exp => @math.exp(x)
      UnaryOp::Ln => if x > 0.0 { @math.ln(x) } else { -1.0 / 0.0 }
      UnaryOp::Sqrt => if x >= 0.0 { x.sqrt() } else { -1.0 / 0.0 }
      UnaryOp::Sin => @math.sin(x)
      UnaryOp::Cos => @math.cos(x)
      UnaryOp::Tanh => @math.tanh(x)
      UnaryOp::Sigmoid => 1.0 / (1.0 + @math.exp(-x))
      UnaryOp::Softplus => {
        let mx = if x > 0.0 { x } else { 0.0 }
        let ax = if x < 0.0 { -x } else { x }
        mx + @math.ln(1.0 + @math.exp(-ax))
      }
      UnaryOp::Relu => if x > 0.0 { x } else { 0.0 }
      UnaryOp::Silu => x / (1.0 + @math.exp(-x))
      UnaryOp::Gelu => {
        let u = gelu_c * (x + 0.044715 * x * x * x)
        0.5 * x * (1.0 + @math.tanh(u))
      }
    }
    out.data[i] = v
  }
  out
}

///|
/// Apply a binary operator elementwise, broadcasting the operands
/// against each other.
pub fn t_binary(op : BinaryOp, a : Tensor, b : Tensor) -> Tensor? {
  let shape = match common_shape(a.shape, b.shape) {
    Some(s) => s
    None => return None
  }
  let x = match a.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let y = match b.broadcast_to(shape) {
    Some(t) => t
    None => return None
  }
  let out = Tensor::zeros(shape)
  for i = 0; i < out.data.length(); i = i + 1 {
    let p = x.data[i]
    let q = y.data[i]
    out.data[i] = match op {
      BinaryOp::Add => p + q
      BinaryOp::Sub => p - q
      BinaryOp::Mul => p * q
      BinaryOp::Div => p / q
    }
  }
  Some(out)
}

///|
/// `base ^ exponent`, elementwise.
pub fn t_pow(a : Tensor, exponent : Double) -> Tensor {
  let out = Tensor::zeros(a.shape)
  for i = 0; i < a.data.length(); i = i + 1 {
    let x = a.data[i]
    out.data[i] = if x == 0.0 && exponent < 0.0 {
      -1.0 / 0.0
    } else {
      @math.pow(x, exponent)
    }
  }
  out
}
