// Tensor basics.
//
// Shows the tensor type on its own terms: shapes, broadcasting,
// matrix products, reductions, and the reshape/transpose pair.
// No differentiation is involved — this is the layer underneath the
// gradient compiler, and it is worth being able to read on its own.

///|
fn main {
  println("| moongrad: tensor basics")
  println("|")
  // A 2x3 matrix and a 3-vector.
  let a = match @moongrad.Tensor::matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) {
    Some(t) => t
    None => {
      println("| the matrix should build")
      return
    }
  }
  let b = @moongrad.Tensor::vector([1.0, 0.5, -1.0])
  println("| a = \{a.preview()}")
  println("| b = \{b.preview()}")
  println("|")
  // Broadcasting: a row vector added to every row of the matrix.
  println("| broadcasting (2,3) + (3,)")
  match @moongrad.Tensor::add_broadcast(a, b) {
    Some(t) => println("|   a + b = \{t.preview()}")
    None => println("|   the broadcast add failed")
  }
  println("|")
  // A column vector added to every column of the matrix. The shape
  // has to be (2, 1): a bare (2,) would try to broadcast against the
  // matrix's *last* axis, and 2 does not match 3.
  println("| broadcasting (2,3) + (2,1)")
  let c = match @moongrad.Tensor::from_data([2, 1], [10.0, 20.0]) {
    Some(t) => t
    None => {
      println("| the column should build")
      return
    }
  }
  match @moongrad.Tensor::add_broadcast(a, c) {
    Some(t) => println("|   a + column = \{t.preview()}")
    None => println("|   the broadcast add failed")
  }
  println("|")
  // Products.
  println("| products")
  match a.matmul_t(a) {
    Some(t) => println("|   a @ a^T = \{t.preview()}")
    None => println("|   the product failed")
  }
  match a.transpose([1, 0]) {
    Some(t) => println("|   a^T = \{t.preview()}")
    None => println("|   the transpose failed")
  }
  println("|")
  // Reductions.
  println("| reductions")
  match @moongrad.t_sum_axis(a, 0) {
    Some(t) => println("|   sum along axis 0 = \{t.preview()}")
    None => println("|   the reduction failed")
  }
  match @moongrad.t_sum_axis(a, 1) {
    Some(t) => println("|   sum along axis 1 = \{t.preview()}")
    None => println("|   the reduction failed")
  }
  println("|   mean of everything = \{@moongrad.format_double(a.mean())}")
  println("|")
  // A rank-4 tensor, the shape a convolutional layer consumes.
  println("| a rank-4 tensor: (batch, channels, height, width)")
  let img = @moongrad.Tensor::filled([2, 3, 4, 5], 0.25)
  println("|   \{img.preview()}")
  println("|   rank = \{img.rank()}, elements = \{img.numel()}")
  println("|")
  println("| The gradient compiler in tensor_grad.mbt works on exactly")
  println("| this representation: one tensor per node, one adjoint")
  println("| tensor per node, and a right-to-left sweep because a node")
  println("| can only be built after its children exist.")
}
