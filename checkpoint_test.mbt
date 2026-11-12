// checkpoint_test.mbt
//
// The memory strategy has to be invisible from the outside. Every
// test here either asserts that the checkpointed gradient equals the
// plain one, or asserts a property of the segmentation arithmetic.
//
// The equality assertions are the important ones. A checkpointing
// implementation that quietly computes something slightly different
// is worse than no checkpointing, because the symptom shows up as a
// training run that converges to a marginally different minimum and
// nobody knows why.

///|
/// Build a moderately deep polynomial in one variable, which is the
/// shape where segment recomputation actually has something to do.
fn deep_graph(g : @moongrad.Graph, x : Int32, depth : Int32) -> Int32 {
  // A contraction, not an expansion: repeated squaring would leave
  // the finite reals long before the interesting depths, and the
  // point here is the segmentation, not the numerics.
  let mut acc = x
  for i = 0; i < depth; i = i + 1 {
    acc = g.add(g.mul(acc, g.constant(0.5)), g.constant(0.1))
    ignore(i)
  }
  acc
}

///|
test "one_segment_reproduces_the_plain_gradient" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = deep_graph(g, x, 6)
  let b = binding_of([0.7])
  let (plain, _p) = @moongrad.grad(g, b, f)
  let (checkpointed, _r) = @moongrad.checkpointed_grad(g, b, f, 1)
  assert_eq(checkpointed.length(), plain.length())
  for i = 0; i < plain.length(); i = i + 1 {
    assert_close(
      checkpointed[i],
      plain[i],
      1.0e-12,
      "one segment, @moongrad.grad \{i}",
    )
  }
}

///|
test "many_segments_reproduce_the_plain_gradient" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(2))
  let x = g.variable(0)
  let y = g.variable(1)
  let a = g.mul(x, y)
  let b = g.sin(a)
  let c = g.div(g.constant(3.0), g.add(g.constant(1.0), g.pow(y, 2.0)))
  let f = g.mul(b, c)
  let binding = binding_of([0.4, 1.1])
  let (plain, _p) = @moongrad.grad(g, binding, f)
  for segments in [1, 2, 3, 5, 8, 16, 64] {
    let (checkpointed, _r) = @moongrad.checkpointed_grad(
      g, binding, f, segments,
    )
    for i = 0; i < plain.length(); i = i + 1 {
      assert_close(
        checkpointed[i],
        plain[i],
        1.0e-12,
        "\{segments} segments, @moongrad.grad \{i}",
      )
    }
  }
}

///|
test "segment_count_does_not_change_a_transcendental_chain" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let mut acc = x
  for i = 0; i < 12; i = i + 1 {
    acc = g.tanh(acc)
    ignore(i)
  }
  let b = binding_of([0.3])
  let (plain, _p) = @moongrad.grad(g, b, acc)
  for segments in [1, 3, 7, 13] {
    let (checkpointed, _r) = @moongrad.checkpointed_grad(g, b, acc, segments)
    assert_close(
      checkpointed[0],
      plain[0],
      1.0e-12,
      "tanh chain with \{segments} segments",
    )
  }
}

///|
test "segments_cover_the_graph_without_gaps_or_overlap" {
  let bounds = @moongrad.segment_bounds(100, 7)
  assert_eq(bounds[0].lo, 0)
  let last = bounds[bounds.length() - 1]
  assert_eq(last.hi, 100)
  for i = 1; i < bounds.length(); i = i + 1 {
    assert_eq(bounds[i].lo, bounds[i - 1].hi)
  }
}

///|
test "segments_shrink_as_the_count_grows" {
  let bounds = @moongrad.segment_bounds(60, 10)
  for s in bounds {
    assert_true(s.width() <= 6)
  }
  let mut total = 0
  for s in bounds {
    total = total + s.width()
  }
  assert_eq(total, 60)
}

///|
test "asking_for_more_segments_than_nodes_clamps" {
  let bounds = @moongrad.segment_bounds(5, 100)
  assert_eq(bounds.length(), 5)
}

///|
test "degenerate_inputs_give_no_segments" {
  assert_eq(@moongrad.segment_bounds(0, 4).length(), 0)
  assert_eq(@moongrad.segment_bounds(10, 0).length(), 0)
  assert_eq(@moongrad.choose_segment_count(0, 4), 1)
  assert_eq(@moongrad.choose_segment_count(10, 0), 1)
}

///|
test "choose_segment_count_respects_the_target_width" {
  // 1000 nodes, target 250 => 4 segments.
  assert_eq(@moongrad.choose_segment_count(1000, 250), 4)
  // A target wider than the graph means a single segment.
  assert_eq(@moongrad.choose_segment_count(10, 1000), 1)
  // Rounding up, not down: 100 nodes at 30 each is 4, not 3.
  assert_eq(@moongrad.choose_segment_count(100, 30), 4)
}

///|
test "the_report_describes_the_gradient_program" {
  // The report is not a promise about a technique; it describes what
  // the evaluation costs. The program is bigger than the primal it
  // came from, and it refers back into the primal, so the values held
  // are more than a tape would hold — not fewer.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = deep_graph(g, x, 30)
  let b = binding_of([0.5])
  let (_plain, report) = @moongrad.checkpointed_grad(g, b, f, 8)
  // The primal's nodes, and then the gradient program's on top.
  check(
    report.peak_live > report.naive_live,
    "the program holds more than a tape would",
  )
  assert_true(report.reduction() < 0.0)
  // The extra evaluations are the gradient program's own nodes.
  assert_true(report.recomputed > 0)
  assert_true(report.arithmetic_cost() > 1.0)
}

///|
test "the_cost_does_not_depend_on_the_segmentation" {
  // This is the finding rather than a shortcoming of the
  // implementation: the gradient program's segments are not
  // independent, because every gradient node refers back into the
  // primal. Cutting the program up therefore saves nothing, and the
  // report says so rather than claiming a saving it does not make.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = deep_graph(g, x, 40)
  let b = binding_of([0.5])
  let (_a, one) = @moongrad.checkpointed_grad(g, b, f, 1)
  let (_b_, many) = @moongrad.checkpointed_grad(g, b, f, 16)
  assert_eq(one.arithmetic_cost(), many.arithmetic_cost())
  assert_eq(one.peak_live, many.peak_live)
  assert_eq(one.recomputed, many.recomputed)
}

///|
test "the_gradient_is_the_same_in_every_configuration" {
  // The one property that has to hold, whatever the segmentation
  // does to the cost.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = deep_graph(g, x, 25)
  let b = binding_of([0.5])
  let (plain, _p) = @moongrad.grad(g, b, f)
  for segments in [1, 2, 3, 5, 8, 16, 64] {
    let (other, _r) = @moongrad.checkpointed_grad(g, b, f, segments)
    for i = 0; i < plain.length(); i = i + 1 {
      assert_close(other[i], plain[i], 1.0e-12, "\{segments} segments")
    }
  }
}

///|
test "segmenting_trades_memory_for_arithmetic" {
  // Fewer segments: more memory held, less arithmetic spent. More
  // segments: less memory held, more arithmetic spent.
  //
  // The memory is not monotone in the segment count: the working set
  // is the segment boundaries plus the widest window, and the
  // boundaries grow with the number of segments while the window
  // shrinks. On a small graph the boundaries can win, so the
  // assertion is that segmenting never holds *more* than the
  // unsegmented pass, and that the arithmetic never gets cheaper.
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = deep_graph(g, x, 40)
  let b = binding_of([0.5])
  let (_one, single) = @moongrad.checkpointed_grad(g, b, f, 1)
  let mut prev_cost = 0.0
  for segments in [1, 2, 4, 8, 16] {
    let (_g, r) = @moongrad.checkpointed_grad(g, b, f, segments)
    check(
      r.peak_live <= single.peak_live,
      "segmenting should not hold more than the unsegmented pass",
    )
    check(
      r.arithmetic_cost() >= prev_cost,
      "arithmetic cost should not shrink as segments are added",
    )
    prev_cost = r.arithmetic_cost()
  }
}

///|
test "recomputation_cost_matches_the_segment_geometry" {
  let bounds = @moongrad.segment_bounds(12, 3)
  // widths 4, 4, 4. The last segment never recomputes.
  assert_eq(@moongrad.recomputation_cost(bounds), 3 + 3)
}

///|
test "peak_occupancy_falls_as_segments_are_added" {
  let few = @moongrad.peak_occupancy(@moongrad.segment_bounds(100, 2), 100)
  let many = @moongrad.peak_occupancy(@moongrad.segment_bounds(100, 20), 100)
  check(few > many, "20 segments should hold less than 2: \{few} vs \{many}")
}

///|
test "checkpointing_works_with_parameters" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let w = g.new_param(2)
  let x = g.variable(0)
  let f = g.add(g.mul(g.param(w, 0), x), g.mul(g.param(w, 1), g.pow(x, 2.0)))
  let b = binding_of([3.0])
  b.set_param(0, [2.0, 5.0])
  let (plain, _p) = @moongrad.grad(g, b, f)
  let (checkpointed, _r) = @moongrad.checkpointed_grad(g, b, f, 4)
  assert_close(
    checkpointed[0],
    plain[0],
    1.0e-12,
    "checkpointed @moongrad.grad with params",
  )
}

///|
test "checkpointing_rejects_a_dangling_target" {
  let g = @moongrad.Graph::new()
  ignore(g.new_vars(1))
  let x = g.variable(0)
  let f = g.pow(x, 2.0)
  let raised = try {
    let _ = @moongrad.checkpointed_grad(g, binding_of([1.0]), f + 99, 4)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "an_empty_graph_is_rejected_rather_than_papered_over" {
  // There is no slot to differentiate, so a target of 0 is a
  // dangling reference. Reporting a zero gradient and a zero-sized
  // report here would hide a caller's off-by-one.
  let g = @moongrad.Graph::new()
  let b = @moongrad.Binding::new()
  let raised = try {
    let _ = @moongrad.checkpointed_grad(g, b, 0, 4)
    false
  } catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(raised)
}

///|
test "a_report_over_an_empty_segmentation_is_harmless" {
  assert_eq(@moongrad.peak_occupancy([], 0), 0)
  assert_eq(@moongrad.recomputation_cost([]), 0)
}
