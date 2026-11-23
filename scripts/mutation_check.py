#!/usr/bin/env python3
"""Check that the tests still catch the mistakes they were written for.

A test that has never failed is not known to work. This script makes
the implementation wrong on purpose, in one place at a time, and
checks that the suite notices. Each mutation below is a change of the
kind that was once a real defect in this library, and every one of
them has a regression test that fails when the mutation is applied.

Run it:

    python3 scripts/mutation_check.py

It needs the MoonBit toolchain on the path, and it takes a few
minutes, because each mutation runs a test file. It is not part of CI
— it is a thing to run when the implementation changes underneath the
tests.

A mutation that leaves the suite green is reported at the end. That is
the interesting outcome: it means a change of that kind could be made
again without anything going red.
"""

import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, file, the text to replace, what replaces it, why it matters,
#  the test file that should go red)
MUTATIONS = [
    (
        "the momentum buffer",
        "optim_tensor.mbt",
        "self.m[idx] = mu * self.m[idx] + g.data[j]",
        "self.m[idx] = g.data[j]",
        "the momentum buffer is not carried into the next step",
        "optim_extra_test.mbt",
    ),
    (
        "adam's bias correction",
        "optim.mbt",
        "  let bias_v = 1.0 - @math.pow(b2, step_t)",
        "  let bias_v = 1.0",
        "adam's second-moment bias correction is not applied",
        "optim_test.mbt",
    ),
    (
        "the tensor adam's bias correction",
        "optim_tensor.mbt",
        "let t = if step_index < 1 { 1.0 } else { step_index.to_double() }",
        "let t = 1.0",
        "the tensor side's bias correction is frozen at the first step",
        "optim_extra_test.mbt",
    ),
    (
        "the learning-rate schedule",
        "optim_extra.mbt",
        "schedule.at(step_index - 1)",
        "schedule.at(0)",
        "the schedule never advances with the step",
        "optim_extra_test.mbt",
    ),
    (
        "adagrad's epsilon",
        "optim_extra.mbt",
        "          self.v[idx] = self.v[idx] + gv * gv\n"
        "          p.data[j] = p.data[j] - lr * gv / (self.v[idx].sqrt() + eps)",
        "          self.v[idx] = self.v[idx] + gv * gv\n"
        "          p.data[j] = p.data[j] - lr * gv / (self.v[idx] + eps).sqrt()",
        "the epsilon sits inside the square root instead of outside it",
        "optim_extra_test.mbt",
    ),
    (
        "the direction slot",
        "optim_second.mbt",
        "if slot < g.input_count() {",
        "if slot < 0 {",
        "the direction slot may take over one of the primal's own inputs",
        "optim_second_test.mbt",
    ),
    (
        "the space of the input product",
        "optim_second.mbt",
        "let own : Array[Tensor] = []\n    for i = 0; i < g.input_count(); i = i + 1 {",
        "let own : Array[Tensor] = b.inputs.copy()\n    for i = 0; i < b.inputs.length(); i = i + 1 {",
        "the product over the inputs counts the direction slot as one "
        "of the primal's inputs",
        "optim_second_test.mbt",
    ),
    (
        "relu's derivative at zero",
        "tensor_grad.mbt",
        "if x.data[i] > 0.0 { 1.0 } else { 0.0 }",
        "if x.data[i] >= 0.0 { 1.0 } else { 0.0 }",
        "relu's derivative at exactly zero is a convention, and the two "
        "engines have to agree on it",
        "engine_agreement_test.mbt",
    ),
    (
        "the constant folder",
        "simplify.mbt",
        "UnaryOp::Tanh => Some(@math.tanh(x))",
        "UnaryOp::Tanh => Some(x)",
        "the constant folder turns tanh into the identity",
        "conventions_test.mbt",
    ),
    (
        "the cholesky factor's diagonal",
        "linalg.mbt",
        "l.mset(i, j, sum.sqrt())",
        "l.mset(i, j, -sum.sqrt())",
        "the cholesky factor's diagonal is negative, which still "
        "reconstructs the matrix",
        "conventions_test.mbt",
    ),
    (
        "the qr factor's sign",
        "linalg.mbt",
        "let alpha = if r.mget(k, k) > 0.0 { -norm } else { norm }",
        "let alpha = if r.mget(k, k) > 0.0 { norm } else { -norm }",
        "the sign of the diagonal of R is a convention, and a "
        "convention that is not written down can change",
        "linalg_test.mbt",
    ),
    (
        "the matrix exponential",
        "linalg.mbt",
        "for i = 1; i <= 16; i = i + 1 {",
        "for i = 1; i <= 6; i = i + 1 {",
        "the series is truncated before it has converged, which the "
        "nilpotent matrices the other tests use cannot see",
        "linalg_test.mbt",
    ),
    (
        "the jacobi sweep budget",
        "linalg.mbt",
        "for _sweep = 0; _sweep < 6; _sweep = _sweep + 1 {",
        "for _sweep = 0; _sweep < 1; _sweep = _sweep + 1 {",
        "the eigenvalues come back as an approximation that is close "
        "enough to be mistaken for the right one",
        "linalg_test.mbt",
    ),
    (
        "the hessian's jacobi sweep budget",
        "hessian.mbt",
        "for _sweep = 0; _sweep < 6; _sweep = _sweep + 1 {",
        "for _sweep = 0; _sweep < 1; _sweep = _sweep + 1 {",
        "the same sweep budget, on the matrix side",
        "hessian_test.mbt",
    ),
    (
        "the matrix norm",
        "matrix.mbt",
        "  acc.sqrt()\n}",
        "  acc\n}",
        "the norm is reported as the sum of the squares",
        "public_api_test.mbt",
    ),
    (
        "the arithmetic cost",
        "checkpoint.mbt",
        "1.0 + self.gradient_evals.to_double() / self.forward_evals.to_double()",
        "1.0 + self.forward_evals.to_double() / self.gradient_evals.to_double()",
        "the two counts in the ratio are the wrong way round",
        "checkpoint_test.mbt",
    ),
    (
        "the tensor trainer's refreshers",
        "train_tensor.mbt",
        "for f in self.refreshers {",
        "for f in [] {",
        "the dropout mask is never redrawn, and a dropout that is not "
        "redrawn is a fixed random projection",
        "train_test.mbt",
    ),
    (
        "the tensor trainer's step index",
        "train_tensor.mbt",
        "self.opt.step(self.params, grads, self.step_count)",
        "self.opt.step(self.params, grads, self.step_count - 1)",
        "the optimiser is told which step it is on, one step too late",
        "train_test.mbt",
    ),
    (
        "the tensor trainer's history",
        "train_tensor.mbt",
        "self.history.push(loss_value)",
        "ignore(loss_value)",
        "the loss history is never recorded, so the first and the last "
        "loss are both zero",
        "train_test.mbt",
    ),
    (
        "the batch trainer's mask",
        "train_batch.mbt",
        "  self.graph.refresh_stochastic(self.binding)",
        "  ignore(0)",
        "the batch trainer does not redraw the mask either",
        "train_batch_test.mbt",
    ),
]


def run(cmd, cwd):
    env = dict(os.environ)
    env.setdefault("HOME", "/workspace/.moonhome")
    env["PATH"] = "/workspace/.moonhome/.moon/bin:" + env.get("PATH", "")
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800
    )


def main():
    work = tempfile.mkdtemp(prefix="mutation-", dir=os.path.dirname(REPO))
    shutil.copytree(
        REPO,
        os.path.join(work, "repo"),
        ignore=shutil.ignore_patterns("_build", ".git"),
    )
    repo = os.path.join(work, "repo")

    baseline = run(["moon", "test", "optim_test.mbt"], repo)
    if "failed: 0" not in baseline.stdout:
        print("the baseline is not green; fix that first")
        print(baseline.stdout[-2000:])
        return 1

    results = []
    for name, path, old, new, why, test_file in MUTATIONS:
        target = os.path.join(repo, path)
        with open(target, encoding="utf-8") as handle:
            original = handle.read()
        if old not in original:
            results.append((name, path, why, "not found", test_file))
            continue
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(original.replace(old, new, 1))
        outcome = run(["moon", "test", test_file], repo)
        text = outcome.stdout + outcome.stderr
        if "failed: 0" in text:
            verdict = "SURVIVED"
        elif "Total tests" in text:
            verdict = "caught"
        else:
            verdict = "compile error"
        results.append((name, path, why, verdict, test_file))
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(original)

    width = max(len(r[0]) for r in results) + 2
    print()
    print(f"{'what was broken'.ljust(width)}{'verdict':<16}why")
    print("-" * (width + 80))
    for name, path, why, verdict, note in results:
        print(f"{name.ljust(width)}{verdict:<16}{why}")
        if verdict in ("SURVIVED", "not found"):
            print(f"{'':<{width}}{'':<16}nothing in {note} noticed")
    survivors = [r for r in results if r[3] in ("SURVIVED", "not found")]
    print(f"\n{len(results)} mutations, {len(survivors)} survived")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
