# Contributing to moongrad

Thanks for looking at this. The notes below are the ones that actually
save time: what the project expects from a change, and how to check it
before you push.

## Toolchain

moongrad is written against **moonc 1.0.0-rc1 or newer**, and it
uses `Int32` as the integer type — the formatter rewrites `Int` to
`Int32`, so please do not hand-edit that back.

```bash
python3 scripts/check_toolchain.py --min 1.0.0-rc1
```

Older toolchains will not build the project. The CI installs the
latest release and fails if the compiler is older than that, so a
change that only builds on a newer toolchain is fine; a change that
needs an older one is not.

## Before you push

```bash
moon fmt              # format in place
moon fmt --check      # what CI runs
moon check            # type check; must be 0 errors, 0 warnings
moon test             # the whole suite
moon build --target all
```

`moon fmt --check` and `moon check` are both enforced in CI, and the
suite has to be green. The CI job finds the examples rather than
listing them, so an example that is added under `examples/` is
smoke-tested without anything having to be edited.

## What a change should look like

- **One thing per commit.** The history is a sequence of small
  steps, each of which builds and passes the tests on its own. A
  commit that fixes one rule and renames a file is two commits.
- **A test that fails without the change.** This is the part that
  matters most. A fix to a derivative rule, a shape adjustment, an
  optimiser's state, or an error path should come with a test that
  reproduces the old behaviour and fails against it. If you cannot
  write that test, the change is not understood well enough yet.
- **A line in the CHANGELOG.** Every user-visible change gets an
  entry, under the version it ships in. The changelog is the project's
  record of what was wrong and what changed; it is worth writing
  honestly, including about the changes that were not obvious.
- **Doc comments on public functions.** `///|` followed by prose.
  Say what the function computes, what it refuses to do, and why when
  the reason is not obvious from the signature.

## Style

The code is meant to be read by someone who did not write it. A few
conventions that the existing code follows:

- Names say what the thing is, not how it is implemented.
  `t_sum_to_shape`, not `reduce_helper_2`.
- Comments explain *why*. What the code does is already in the code.
- Failures are refused rather than guessed at: an operation that
  cannot be done returns `None` or raises a `suberror` that carries
  the operands that caused it. It does not return a zero and hope.
- No `println` in the library. The examples print; the library does
  not.

## Pull requests

Keep the description short and specific: what was wrong, what changed,
and how the change was checked. If the change touches a derivative
rule, say which function's gradient changed and what the new gradient
is.

If you are fixing something that was reported, reference the issue.
If the change is large, please open an issue first so the approach can
be agreed before the work is done.
