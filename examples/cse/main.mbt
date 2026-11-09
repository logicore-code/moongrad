// What the compiler saves.
//
// Builds the same expression twice — once written out so that every
// operation is a separate builder call, once relying on the
// compiler to notice that two of them are the same — and prints how
// many slots each needed.
//
// The point is not that the naive version is slow. The point is that
// the *compiler* does the sharing, not the user. A caller writes
// `x*y` twice because the algebra says to, and the builder hands
// back the same slot both times.
//
// The example also shows the peephole rules firing. `x + 0` and
// `x * 1` collapse to `x`; `x * 0` collapses to `0` and takes the
// whole subtree with it; and a subtree whose leaves are all
// literals is evaluated at build time.

///|
fn main raise @moongrad.ADError {
  println("| moongrad: what the graph compiler saves")
  println("|")

  // A builder call per operation, with the repeated product
  // written out twice.
  let naive = @moongrad.Graph::new()
  ignore(naive.new_vars(2))
  let nx = naive.variable(0)
  let ny = naive.variable(1)
  let a = naive.mul(nx, ny)
  let b = naive.mul(nx, ny)
  let c = naive.add(a, b)
  let d = naive.mul(c, c)
  let _ = naive.mul(d, d)
  println("| (x*y + x*y)^2, written out:")
  println("|   builder calls = \{naive.stats.requests}")
  println("|   slots used    = \{naive.len()}")
  println("|   shared away   = \{naive.stats.shared_away()}")
  println("|")

  // The same expression, but the shared product is computed once
  // and referenced twice. The compiler cannot tell these apart —
  // that is the point.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let p = g.mul(x, y)
  let q = g.add(p, p)
  let r = g.mul(q, q)
  let _ = g.mul(r, r)
  println("| the same expression, sharing the product explicitly:")
  println("|   builder calls = \{g.stats.requests}")
  println("|   slots used    = \{g.len()}")
  println("|   shared away   = \{g.stats.shared_away()}")
  println("|")
  println("| Same slot count either way. The compiler gave back the")
  println("| same slot for the second x*y whether or not the caller")
  println("| asked for it.")
  println("|")

  // The simplifier's rules, one at a time, so each is visible.
  let g2 = @moongrad.Graph::new()
  ignore(g2.new_vars(1))
  let z = g2.variable(0)
  let zero = g2.constant(0.0)
  let one = g2.constant(1.0)
  let twice = g2.mul(z, g2.constant(2.0))
  let cases : Array[(String, Int32, Int32)] = []
  cases.push(("x + 0", z, g2.add(z, zero)))
  cases.push(("x * 1", z, g2.mul(z, one)))
  cases.push(("0 * x", z, g2.mul(zero, z)))
  cases.push(("x - x", z, g2.sub(z, z)))
  cases.push(
    (
      "(x*2) + (x*2)",
      twice,
      g2.add(g2.mul(z, g2.constant(2.0)), g2.mul(z, g2.constant(2.0))),
    ),
  )
  println("| rule                result")
  println("| ---------------------------")
  for c in cases {
    let (name, expected, got) = c
    let same = if expected == got { "-> same slot as x" } else { "-> changed" }
    println("| \{name}   \{same}")
  }
  println("|")
  // Constant folding: a subtree whose leaves are all literals is
  // evaluated while the graph is being built.
  let g3 = @moongrad.Graph::new()
  let a = g3.constant(2.0)
  let b = g3.constant(3.0)
  let folded = g3.add(g3.mul(a, b), a)
  println("| constant folding: (2*3) + 2 = \{g3.eval_constant(folded)}")
  println("|   folds performed = \{g3.stats.folds}, slots = \{g3.len()}")
  println("|")
  println("| Every rule is exact in IEEE-754 for the operands it")
  println("| accepts, and nothing is rewritten when an operand is")
  println("| non-finite. A simplifier that changes the rounding of the")
  println("| caller's program is worse than no simplifier, because the")
  println("| symptom is a training run that converges to a slightly")
  println("| different minimum and nobody can say why.")
}
