// A gradient, computed in the browser.
//
// The program is deliberately small: it compiles a gradient and
// evaluates it, and the point is not the arithmetic but where it
// runs. The same source builds for wasm, for wasm-gc, for JavaScript
// and for native, and this one is meant to be opened in a tab.
//
// The output goes to stdout, which the page captures through the
// WebAssembly system interface. There is no JavaScript glue beyond
// that shim: no FFI, no runtime, no second implementation of
// anything.

///|
fn main raise @moongrad.ADError {
  println("| moongrad, running where the page is running it")
  println("|")
  // f(x) = x^2 * exp(x), differentiated with respect to x.
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  b.set_input(0, @moongrad.Tensor::vector([0.7]))
  let x = g.variable(0)
  let square = g.mul(x, x)
  let f = g.mul(square, g.exp(x))
  let grads = @moongrad.t_grad(g, b, f)
  let d = grads.input_grads[0].data[0]
  println("| f(x) = x^2 * exp(x) at x = 0.7")
  println("| df/dx = 2x*exp(x) + x^2*exp(x) = \{d}")
  println("|")
  // The same program, evaluated at a second point, which is the
  // property the design exists for: the gradient is a program, and a
  // program can be run again on different numbers.
  b.set_input(0, @moongrad.Tensor::vector([1.3]))
  let grads2 = @moongrad.t_grad(g, b, f)
  println(
    "| at x = 1.3 the same gradient program gives \{grads2.input_grads[0].data[0]}",
  )
}
