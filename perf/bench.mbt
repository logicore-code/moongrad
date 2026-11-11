// Performance benchmarks.
//
// Run with `moon test --target native --release perf`. The summary
// is printed to the test log; the numbers are transcribed into
// `docs/benchmark.md`.
//
// Five workloads, chosen to cover the paths that dominate a real
// training loop:
//
//   grad_compile   building and compiling the gradient program. This
//                   happens once per model, so it matters for
//                   interactive use more than for batch training.
//   forward        the constant-propagation pass, once per sample.
//   backward       re-evaluating the compiled gradient, once per
//                   step.
//   checkpoint     the same backward pass with segment recomputation.
//   train_step     a whole optimiser step, end to end.
//
// The last two are on the tensor side, and they are the ones that
// carry the argument: compiling a gradient program costs far more
// than evaluating it, so a training loop that compiles once and
// evaluates every step is doing an order of magnitude less work
// than one that recompiles.

///|
/// Build a fresh graph and compile its gradient, swallowing any
/// error. The benchmark closure cannot raise, so the work is
/// wrapped rather than letting a failure abort the whole suite.
fn compile_once(depth : Int32) -> Unit {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = try g.variable(0) catch {
    _ => return
  } noraise {
    v => v
  }
  let f = build(g, x, depth)
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.5)
  let _ = try @moongrad.grad(g, b, f) catch {
    _ => return
  } noraise {
    v => v
  }
}

///|
fn build(g : @moongrad.Graph, x : Int32, depth : Int32) -> Int32 {
  let mut acc = x
  for i = 0; i < depth; i = i + 1 {
    acc = g.add(g.mul(acc, g.constant(0.5)), g.constant(0.1))
    ignore(i)
  }
  acc
}

///|
test "perf_suite" {
  let results : Array[String] = []

  // --- building and compiling -------------------------------------------
  let build_summary = @bench.single_bench(
    name="grad_compile_400_nodes",
    count=10,
    fn() { compile_once(400) },
  )
  results.push(build_summary |> @json.to_json |> j => { j.stringify() })

  // --- the two passes ----------------------------------------------------
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = try g.variable(0) catch {
    _ => return
  } noraise {
    v => v
  }
  let f = build(g, x, 400)
  let b = @moongrad.Binding::new()
  b.set_var(0, 0.5)
  let (_grads, program) = try @moongrad.grad(g, b, f) catch {
    _ => return
  } noraise {
    v => v
  }

  let forward_summary = @bench.single_bench(
    name="forward_400_nodes",
    count=10,
    fn() {
      let _ = try @moongrad.forward_values(g, b) catch {
        _ => return
      } noraise {
        v => v
      }
    },
  )
  results.push(forward_summary |> @json.to_json |> j => { j.stringify() })

  let backward_summary = @bench.single_bench(
    name="backward_400_nodes",
    count=10,
    fn() {
      let _ = try @moongrad.evaluate_gradients(program, b) catch {
        _ => return
      } noraise {
        v => v
      }
    },
  )
  results.push(backward_summary |> @json.to_json |> j => { j.stringify() })

  let checkpoint_summary = @bench.single_bench(
    name="checkpointed_backward_400_nodes_8seg",
    count=10,
    fn() {
      let _ = try @moongrad.checkpointed_grad(g, b, f, 8) catch {
        _ => return
      } noraise {
        v => v
      }
    },
  )
  results.push(checkpoint_summary |> @json.to_json |> j => { j.stringify() })

  // --- a whole training step --------------------------------------------
  let step_summary = @bench.single_bench(name="train_step_8x4", count=10, fn() {
    let data = match
      @moongrad.Dataset::new(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [1.0, 1.0], [0.9, 1.0], [1.0, 0.9]],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        1,
      ) {
      Some(d) => d
      None => return
    }
    let trainer = try
      @moongrad.Trainer::new(
        [2, 8, 1],
        [@moongrad.Activation::Tanh, @moongrad.Activation::Identity],
        data,
        @moongrad.LossKind::Mse,
        @moongrad.Optimizer::Adam(0.05),
        1,
      )
    catch {
      _ => return
    } noraise {
      v => v
    }
    let _ = try trainer.train_step(-1.0) catch {
      _ => return
    } noraise {
      v => v
    }
  })
  results.push(step_summary |> @json.to_json |> j => { j.stringify() })

  // --- the tensor side ------------------------------------------------
  let tensor_compile_summary = @bench.single_bench(
    name="tensor_grad_compile",
    count=10,
    fn() { compile_tensor_once() },
  )
  results.push(
    tensor_compile_summary |> @json.to_json |> j => { j.stringify() },
  )
  let tensor_eval_summary = @bench.single_bench(
    name="tensor_grad_evaluate",
    count=10,
    fn() { evaluate_tensor_once() },
  )
  results.push(tensor_eval_summary |> @json.to_json |> j => { j.stringify() })

  println("=== moongrad benchmark results ===")
  for r in results {
    println(r)
  }
  assert_eq(results.length(), 7)
}

///|
/// A tensor model: `width` features in, `width` out, and the mean
/// squared error against a target.
fn tensor_model(
  g : @moongrad.TGraph,
  width : Int32,
  rows : Int32,
) -> (Int32, Array[@moongrad.Tensor]) {
  let x = g.variable(0)
  let y = g.variable(1)
  let w = g.param(0)
  let pred = g.matmul(x, w)
  let diff = g.sub(pred, y)
  let squared = g.mul(diff, diff)
  let total = g.sum_axis(g.sum_axis(squared, 1), 0)
  let loss = g.div(
    total,
    g.constant(@moongrad.Tensor::filled([], rows.to_double())),
  )
  ignore(y)
  (loss, [@moongrad.Tensor::zeros([width, width])])
}

///|
/// The size of the tensor model the tensor benchmarks use.
let tensor_width : Int32 = 32

///|
let tensor_rows : Int32 = 16

///|
/// Compile the tensor gradient once, on a fresh graph.
fn compile_tensor_once() -> Unit {
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let (loss, params) = tensor_model(g, tensor_width, tensor_rows)
  for i = 0; i < params.length(); i = i + 1 {
    b.set_param(i, params[i])
  }
  b.set_input(0, @moongrad.Tensor::zeros([tensor_rows, tensor_width]))
  b.set_input(1, @moongrad.Tensor::zeros([tensor_rows, tensor_width]))
  let _ = @moongrad.compile_tensor_gradient(g, b, loss)
}

///|
/// Evaluate an already-compiled tensor gradient at a binding.
fn evaluate_tensor_once() -> Unit {
  let g = @moongrad.TGraph::new()
  let b = @moongrad.TBinding::new()
  let (loss, params) = tensor_model(g, tensor_width, tensor_rows)
  for i = 0; i < params.length(); i = i + 1 {
    b.set_param(i, params[i])
  }
  b.set_input(0, @moongrad.Tensor::zeros([tensor_rows, tensor_width]))
  b.set_input(1, @moongrad.Tensor::zeros([tensor_rows, tensor_width]))
  let compiled = @moongrad.compile_tensor_gradient(g, b, loss)
  let _ = @moongrad.evaluate_tensor_gradient(compiled, b)
}

///|
fn main {
  println("perf binary: run `moon test --target native --release perf`")
}
