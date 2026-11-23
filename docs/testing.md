# How the tests are put together

The suite is 518 cases. That number says how much there is, not how much
it checks, so this page is about the second question.

## The rules the suite follows

**Every fix comes with a test that fails before it.** Not a test that
would fail if the bug came back in a slightly different shape — the
test that fails on the code as it was before the fix. This is the rule
that catches the fixes that were never really verified. Several rounds
of this project found that a fix without such a test had left the
original defect in place somewhere else, and the only thing that found
it was the next one.

**The same function is checked by two independent routes.** A gradient
is compared against a central difference of the same function
(`examples/gradcheck`, and the `gradcheck` test file); the tensor
engine's derivative of every operator is compared against the scalar
engine's (`engine_agreement_test.mbt`); the dense Hessian is compared
against the matrix-free Hessian-vector product (`second_order` and
`examples/second_order`); the one-shot product is compared against the
compiled one. A rule that is wrong on one side only cannot pass both
comparisons, and a rule that is wrong on both sides is still caught by
the finite difference, because the finite difference knows nothing
about the rules.

**A failure is reported, not swallowed.** `core_test.mbt` collects the
failures of a whole file into a list instead of stopping at the first
one, because the thing under test is usually a chain rule and the
first broken link hides the rest. The list is reported from
`aaa_summary_test.mbt`, which has to be the last file to run: MoonBit
runs the test files in reverse alphabetical order, and the file named
`aaa_...` is the last one. A summary that ran earlier would miss the
failures of the files after it — and a suite that cannot see its own
failures is worse than one that has none.

**The public surface is exercised.** `public_api_test.mbt` calls every
public function that no other test in the repository calls — the ones
that had been exported and never called, which is how a function ends
up exported and never checked. There are seventy-five of them, and the
file reaches all seventy-five: the list comes from the toolchain's own
generated interface, and every name in it is called from here and from
nowhere else.

## What breaking the implementation on purpose found

Reading the code finds the defects you can see. It does not find the
defects the tests do not cover. So the implementation was broken on
purpose, one place at a time, and the suite was run to see whether it
noticed.

Two rounds were done: sixty-four mutations in the scalar engine and
thirty-four in the tensor engine. The mutations were the kind that a
careless edit produces — a sign flipped, an index off by one, a state
that is read but not carried, a bound that is `<=` where it should be
`<`, a counter that is incremented in the wrong place.

Most were caught. The survivors were the interesting part, because
each one named a path that no test walked:

- **A whole operator was untested.** The graph-level convolution
  appeared in zero test files. The host-side convolution had its own
  tests, and they passed, and they tested a different code path: the
  bias was broadcast over the wrong axis and nothing went red.
- **Only relative behaviour was checked.** The trainer's test asserted
  that the loss fell. Halving the loss leaves "the loss fell" true.
- **An algebraic identity was checked, a convention was not.** A
  Cholesky factorisation with a negative diagonal still satisfies
  `L Lᵀ = A`, so the reconstruction test passed and the factor was not
  a Cholesky factor. The convention — the diagonal is positive — was
  what needed testing.
- **A constant was folded to the wrong value.** The scalar engine's
  constant folder turned `tanh(-1)` into `-1`; the true value is
  −0.7615941559557649. The tensor engine folded it correctly. No test
  compared the two engines on a folded constant, so the disagreement was
  invisible until the implementation was broken on purpose.

Every survivor became a regression test. That is the point of the
exercise: the mutation is not the goal, the test that the mutation
showed was missing is.

## The check is in the repository

`scripts/mutation_check.py` runs the twenty mutations that matter —
the ones a careless edit produces — against a copy of the library, one
at a time, and reports the ones the suite does not notice. It is not
part of CI, because it takes a few minutes; it is a thing to run when
the implementation changes underneath the tests.

```
python3 scripts/mutation_check.py
```

The twenty cover the optimisers' state between steps, the trainers'
state between steps, and the conventions the library documents — the
sign of a Cholesky factor's diagonal, the place Adagrad's epsilon
sits, the sweep budget of the Jacobi decomposition, the length of the
matrix exponential's series. Every one of them has a test that fails
against the code with the convention broken, and the script is how
that is checked rather than asserted.

## A failure has to be visible where it is found

The failures of a test are collected in a shared list and reported
from a summary at the end of the suite, because a gradient is a chain
rule and the first broken link hides the rest. The cost of that
arrangement is that the summary is the only place a failure shows up,
and the summary runs last — so a developer who ran one test file while
working on one rule saw a green run for a file whose tests had failed.

The failure is raised where it is found now, as well as recorded. The
list is still there, and the summary still reports it; the difference
is that the test that found the failure is the one that goes red. The
mutation script depends on this: it runs one test file at a time, and
before the fix it could not tell a caught mutation from a missed one.

## What this does not claim

The suite is not a proof. It is a set of checks that have each failed
at least once, and the mutations above were a systematic way of asking
which parts of the code no check had ever walked.
